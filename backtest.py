from __future__ import annotations

import argparse
import json
import time
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from strategy import compute_indicators, generate_signal

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})


def _to_ms(value: str | None) -> int | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def fetch_ohlcv_full(symbol: str, timeframe: str, since_ms: int | None = None, until_ms: int | None = None) -> pd.DataFrame:
    rows = []
    since = since_ms
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not chunk:
            break
        rows.extend(chunk)
        since = chunk[-1][0] + 1
        if len(chunk) < 1000:
            break
        if until_ms is not None and since >= until_ms:
            break
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if since_ms is not None:
        df = df[df["timestamp"] >= pd.to_datetime(since_ms, unit="ms", utc=True)]
    if until_ms is not None:
        df = df[df["timestamp"] <= pd.to_datetime(until_ms, unit="ms", utc=True)]
    return compute_indicators(df.reset_index(drop=True))


def run_smoke_backtest(symbol: str, timeframe: str, start: str | None = None, end: str | None = None, limit: int = 1000):
    since_ms = _to_ms(start)
    until_ms = _to_ms(end)
    if since_ms is None:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            raise RuntimeError("no market data returned")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = compute_indicators(df)
        df = compute_indicators(df)
    else:
        df = fetch_ohlcv_full(symbol, timeframe, since_ms, until_ms)
    if df.empty or len(df) < 80:
        raise RuntimeError("not enough data for a smoke backtest")

    capital = 10_000.0
    cash = capital
    position: Optional[dict] = None
    trades = []
    equity_curve = []

    for i in range(60, len(df) - 1):
        window = df.iloc[: i + 1].copy().reset_index(drop=True)
        bar = df.iloc[i + 1]

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

        signal = generate_signal(window)
        if position is None and signal and signal.side == "LONG" and signal.strategy != "no_trade":
            entry = float(bar["open"])
            qty = (cash * 0.33) / entry
            position = {"entry": entry, "qty": qty, "sl": entry * (1 - float(signal.stop_loss_pct)), "tp": entry * (1 + float(signal.take_profit_pct))}
            cash -= qty * entry

        equity_curve.append(cash + (position["qty"] * bar["close"] if position else 0.0))

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
        "final_equity": cash,
        "return_pct": (cash / capital - 1) * 100,
        "max_drawdown_pct": float(dd.min() * 100),
        "trades_detail": trades,
    }


def main():
    ap = argparse.ArgumentParser(description="V7 smoke backtest using the live strict trend signal.")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    result = run_smoke_backtest(args.symbol, args.timeframe, start=args.start, end=args.end, limit=args.limit)
    print(json.dumps(result, indent=2, default=str))
    if args.save:
        with open("backtest_result_v7.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
