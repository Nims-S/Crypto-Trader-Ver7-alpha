from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import backtest as bt

RESULT_DIR = Path("backtest_results")
RESULT_DIR.mkdir(exist_ok=True)


def _normalize_regimes(regimes: Iterable[str] | None) -> tuple[str, ...]:
    if not regimes:
        return ("all",)
    cleaned = []
    for r in regimes:
        r = (r or "").strip().lower()
        if not r:
            continue
        cleaned.append(r)
    return tuple(cleaned or ["all"])


def _patch_signal(allowed: set[str]):
    original_signal = bt.signal

    def filtered_signal(symbol, df, p):
        sig = original_signal(symbol, df, p)
        if not sig:
            return None
        if "all" in allowed:
            return sig
        # FIX: filter by regime, not strategy name
        if sig.get("regime") in allowed:
            return sig
        return None

    return original_signal, filtered_signal


def run_one(mode: str, regimes: tuple[str, ...], symbols: list[str], timeframe: str, start: str, end: str, train_start: str, train_end: str, test_start: str, test_end: str, capital: float, maker_fee_bps: float, taker_fee_bps: float, slippage_bps: float, slippage_atr_mult: float, latency_bars: int, trials: int, seed: int, no_cache: bool):
    allowed = set(regimes)
    original_signal, filtered_signal = _patch_signal(allowed)
    bt.signal = filtered_signal
    try:
        cfg = bt.BacktestConfig(
            initial_capital=capital,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
            slippage_atr_mult=slippage_atr_mult,
            latency_bars=latency_bars,
            timeframe=timeframe,
            symbols=tuple(symbols),
            seed=seed,
        )

        if mode == "backtest":
            data = {s: bt.load_history(s, timeframe, start, end, cache=not no_cache) for s in symbols}
            result = bt.Engine(cfg, bt.SignalParams()).run(data)
            payload = {
                "regimes": list(regimes),
                "mode": mode,
                "result": {
                    "total_trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "total_pnl": result.total_pnl,
                    "return_pct": result.return_pct,
                    "max_dd_pct": result.max_drawdown_pct,
                    "pf": result.profit_factor,
                    "sharpe": result.sharpe,
                    "calmar": result.calmar,
                    "by_strategy": result.by_strategy,
                    "by_symbol": result.by_symbol,
                },
            }
            print(json.dumps(payload, indent=2, default=str))
            return payload

        data = {s: bt.load_history(s, timeframe, train_start, test_end, cache=not no_cache) for s in symbols}

        if mode == "optimize":
            train = {
                s: bt.prep(df[(df["timestamp"] >= bt.utc(train_start)) & (df["timestamp"] <= bt.utc(train_end))].reset_index(drop=True))
                for s, df in data.items()
            }
            best_params, train_result, leaderboard = bt.optimize(train, cfg, trials, seed)
            payload = {
                "regimes": list(regimes),
                "mode": mode,
                "best_params": asdict(best_params),
                "train": {
                    "total_trades": train_result.total_trades,
                    "win_rate": train_result.win_rate,
                    "total_pnl": train_result.total_pnl,
                    "return_pct": train_result.return_pct,
                    "max_dd_pct": train_result.max_drawdown_pct,
                    "pf": train_result.profit_factor,
                    "sharpe": train_result.sharpe,
                    "calmar": train_result.calmar,
                },
                "leaderboard_top5": leaderboard[:5],
            }
            print(json.dumps(payload, indent=2, default=str))
            return payload

        p, train_result, test_result, leaderboard = bt.walkforward(
            data, cfg, train_start, train_end, test_start, test_end, trials, seed
        )
        payload = {
            "regimes": list(regimes),
            "mode": mode,
            "best_params": asdict(p),
            "train": {
                "total_trades": train_result.total_trades,
                "win_rate": train_result.win_rate,
                "total_pnl": train_result.total_pnl,
                "return_pct": train_result.return_pct,
                "max_dd_pct": train_result.max_drawdown_pct,
                "pf": train_result.profit_factor,
                "sharpe": train_result.sharpe,
                "calmar": train_result.calmar,
            },
            "test": {
                "total_trades": test_result.total_trades,
                "win_rate": test_result.win_rate,
                "total_pnl": test_result.total_pnl,
                "return_pct": test_result.return_pct,
                "max_dd_pct": test_result.max_drawdown_pct,
                "pf": test_result.profit_factor,
                "sharpe": test_result.sharpe,
                "calmar": test_result.calmar,
            },
            "leaderboard_top5": leaderboard[:5],
        }
        print(json.dumps(payload, indent=2, default=str))
        return payload
    finally:
        bt.signal = original_signal


def main():
    ap = argparse.ArgumentParser(description="Run regime-specific backtest sweeps.")
    ap.add_argument("--mode", choices=["backtest", "optimize", "walkforward"], default="walkforward")
    ap.add_argument("--regimes", nargs="*", default=["all"], help="Choose one or more: trend breakout range chop all")
    ap.add_argument("--symbols", nargs="+", default=bt.DEFAULT_SYMBOLS)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--train-start", default="2022-01-01")
    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--test-end", default="2026-12-31")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--maker-fee-bps", type=float, default=2.0)
    ap.add_argument("--taker-fee-bps", type=float, default=6.0)
    ap.add_argument("--slippage-bps", type=float, default=3.0)
    ap.add_argument("--slippage-atr-mult", type=float, default=0.10)
    ap.add_argument("--latency-bars", type=int, default=1)
    ap.add_argument("--trials", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    regimes = _normalize_regimes(args.regimes)
    sweep = regimes if "all" not in regimes and len(regimes) > 1 else (regimes,)

    outputs = []
    for rset in sweep:
        out = run_one(
            args.mode,
            rset,
            args.symbols,
            args.timeframe,
            args.start,
            args.end,
            args.train_start,
            args.train_end,
            args.test_start,
            args.test_end,
            args.capital,
            args.maker_fee_bps,
            args.taker_fee_bps,
            args.slippage_bps,
            args.slippage_atr_mult,
            args.latency_bars,
            args.trials,
            args.seed,
            args.no_cache,
        )
        outputs.append(out)

    if args.save:
        outfile = RESULT_DIR / f"regime_sweep_{args.mode}.json"
        outfile.write_text(json.dumps(outputs, indent=2, default=str))
        print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
