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


# ===================== CONFIG =====================
TAKER_FEE_BPS = 6.0
MAKER_FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
SLIPPAGE_ATR_MULT = 0.1


def _to_ms(value):
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _sig(signal, key, default=None):
    if signal is None:
        return default
    return getattr(signal, key, default)


def _slippage(price, atr_pct, side):
    slip = (SLIPPAGE_BPS / 10000) + (atr_pct * SLIPPAGE_ATR_MULT)
    if side == "LONG":
        return price * (1 + slip)
    else:
        return price * (1 - slip)


def fetch_ohlcv_full(symbol, timeframe, since_ms=None, until_ms=None):
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
        if until_ms and since >= until_ms:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    return compute_indicators(df.reset_index(drop=True))


def run_backtest(symbol, timeframe, start=None, end=None):
    df = fetch_ohlcv_full(symbol, timeframe, _to_ms(start), _to_ms(end))

    capital = 10000
    cash = capital
    position = None
    trades = []
    equity = []

    cooldown_until = -1

    for i in range(80, len(df) - 1):
        window = df.iloc[: i + 1]
        bar = df.iloc[i + 1]
        entry_idx = i + 1

        # exit
        if position:
            hit_sl = bar["low"] <= position["sl"]
            hit_tp = bar["high"] >= position["tp"]

            if hit_sl or hit_tp:
                exit_price = position["sl"] if hit_sl else position["tp"]
                exit_price = _slippage(exit_price, bar["atr_pct"], position["side"])

                fee = exit_price * position["qty"] * (MAKER_FEE_BPS / 10000)
                pnl = (exit_price - position["entry"]) * position["qty"] - fee

                cash += position["qty"] * exit_price
                trades.append(pnl)

                cooldown_until = entry_idx + position.get("cooldown", 0)
                position = None

        # entry
        signal = generate_signal(window)

        if position is None and signal and entry_idx >= cooldown_until:
            side = _sig(signal, "side")
            entry_price = _slippage(bar["open"], bar["atr_pct"], side)

            qty = (cash * 0.3) / entry_price
            fee = entry_price * qty * (TAKER_FEE_BPS / 10000)

            sl = entry_price * (1 - _sig(signal, "stop_loss_pct"))
            tp = entry_price * (1 + _sig(signal, "take_profit_pct"))

            position = {
                "entry": entry_price,
                "qty": qty,
                "sl": sl,
                "tp": tp,
                "side": side,
                "cooldown": _sig(signal, "cooldown_bars", 0),
            }

            cash -= qty * entry_price + fee

        equity.append(cash + (position["qty"] * bar["close"] if position else 0))

    return {
        "trades": len(trades),
        "profit_factor": sum([t for t in trades if t > 0]) / abs(sum([t for t in trades if t < 0]) or 1),
        "final_equity": cash,
        "return_pct": (cash / capital - 1) * 100,
        "max_drawdown_pct": float((np.array(equity) - np.maximum.accumulate(equity)).min() / capital * 100),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()

    result = run_backtest(args.symbol, args.timeframe, args.start, args.end)
    print(json.dumps(result, indent=2))
