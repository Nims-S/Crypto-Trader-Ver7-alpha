from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from strategy import compute_indicators, generate_signal


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return compute_indicators(df)


def run_smoke_backtest(symbol: str, timeframe: str, limit: int = 1000):
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df.empty or len(df) < 80:
        raise RuntimeError("not enough data for a smoke backtest")

    capital = 10_000.0
    cash = capital
    position: Optional[dict] = None
    trades = []
    equity_curve = []

    # Use closed candles only.
    for i in range(60, len(df) - 1):
        window = df.iloc[: i + 1].copy().reset_index(drop=True)
        bar = df.iloc[i + 1]

        # manage open position on next candle's intrabar range
        if position is not None:
            if bar["low"] <= position["sl"]:
                exit_price = position["sl"]
                pnl = (exit_price - position["entry"]) * position["qty"]
                cash += position["qty"] * exit_price
                trades.append({"entry": position["entry"], "exit": exit_price, "pnl": pnl, "reason": "sl"})
                position = None
            elif bar["high"] >= position["tp"]:
                exit_price = position["tp"]
                pnl = (exit_price - position["entry"]) * position["qty"]
                cash += position["qty"] * exit_price
                trades.append({"entry": position["entry"], "exit": exit_price, "pnl": pnl, "reason": "tp"})
                position = None

        signal = generate_signal(symbol, window)
        if position is None and signal and signal.side == "LONG" and signal.strategy != "no_trade":
            entry = float(bar["open"])
            qty = (cash * 0.33) / entry
            position = {
                "entry": entry,
                "qty": qty,
                "sl": entry * (1 - float(signal.stop_loss_pct)),
                "tp": entry * (1 + float(signal.take_profit_pct)),
                "signal": signal,
            }
            cash -= qty * entry

        equity = cash + (position["qty"] * bar["close"] if position else 0.0)
        equity_curve.append(equity)

    if position is not None:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        pnl = (exit_price - position["entry"]) * position["qty"]
        cash += position["qty"] * exit_price
        trades.append({"entry": position["entry"], "exit": exit_price, "pnl": pnl, "reason": "eod"})

    pnls = np.array([t["pnl"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = int((pnls > 0).sum())
    losses = int((pnls <= 0).sum())
    gross_profit = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if len(pnls) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    final_equity = cash
    return_pct = (final_equity / capital - 1) * 100
    peak = np.maximum.accumulate(np.array(equity_curve, dtype=float)) if equity_curve else np.array([capital])
    dd = (np.array(equity_curve, dtype=float) - peak) / np.where(peak == 0, 1, peak) if equity_curve else np.array([0.0])

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": len(df),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) if trades else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": pf,
        "final_equity": final_equity,
        "return_pct": return_pct,
        "max_drawdown_pct": float(dd.min() * 100),
        "trades_detail": trades,
    }


def main():
    ap = argparse.ArgumentParser(description="V7 smoke backtest using the live strict trend signal.")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    result = run_smoke_backtest(args.symbol, args.timeframe, args.limit)
    print(json.dumps(result, indent=2, default=str))

    if args.save:
        with open("backtest_result_v7.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
