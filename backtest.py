from __future__ import annotations

import argparse
import json
import time

import ccxt
import numpy as np
import pandas as pd

from strategy import StrategyState, compute_indicators, generate_signal

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})

TAKER_FEE_BPS = 6.0
MAKER_FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE = 0.005  # 0.5% of equity per trade
MAX_NOTIONAL_FRAC = 0.25
MAX_BARS_IN_TRADE = 24


def _to_ms(v):
    if not v:
        return None
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _sig(s, k, d=None):
    return getattr(s, k, d) if s is not None else d


def _slip(p, atr, c, side):
    atr_pct = (atr / c) if c else 0
    sl = (SLIPPAGE_BPS / 10000) + (atr_pct * SLIPPAGE_ATR_MULT)
    return p * (1 + sl) if side == "LONG" else p * (1 - sl)


def fetch_ohlcv_full(sym, tf, since=None, until=None):
    rows = []
    cur = since
    while True:
        chunk = exchange.fetch_ohlcv(sym, timeframe=tf, since=cur, limit=1000)
        if not chunk:
            break
        rows.extend(chunk)
        cur = chunk[-1][0] + 1
        if len(chunk) < 1000 or (until and cur >= until):
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Precompute indicators ONCE on the full df (not per-bar)
    df = compute_indicators(df.reset_index())

    # 200 EMA for regime filter: computed causally.
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df["ema200"] = df["close"].ewm(span=200, min_periods=1).mean()

    return df


def _close_partial(cash, entry, exit_price, qty, side, fee_bps=MAKER_FEE_BPS):
    fee = exit_price * qty * (fee_bps / 10000)
    if side == "LONG":
        pnl = (exit_price - entry) * qty - fee
    else:
        pnl = (entry - exit_price) * qty - fee
    cash += qty * exit_price
    return cash, pnl


def run_backtest(sym, tf, start=None, end=None):
    df = fetch_ohlcv_full(sym, tf, _to_ms(start), _to_ms(end))

    cap = 10_000.0
    cash = cap
    pos = None
    trades = []
    eq = []
    cool = -1
    state = StrategyState()

    for i in range(200, len(df) - 1):
        bar = df.iloc[i + 1]
        idx = i + 1

        # Mark-to-market equity (used for risk sizing)
        equity = cash + (pos["qty_open"] * df.iloc[i]["close"] if pos else 0.0)

        # ── Manage open trade ────────────────────────────────────────────────
        if pos:
            pos["bars"] += 1

            # Time stop: close whatever remains at market after max bars
            if pos["bars"] >= MAX_BARS_IN_TRADE:
                ex = _slip(bar["close"], bar["atr"], bar["close"], pos["side"])
                cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], pos["side"], fee_bps=MAKER_FEE_BPS)
                trades.append({
                    "ts": bar.name,
                    "side": pos["side"],
                    "entry": round(pos["entry"], 2),
                    "exit": round(ex, 2),
                    "pnl": round(pnl, 4),
                    "result": "TIME_STOP",
                    "fraction": round(pos["qty_open"] / pos["qty_total"], 4),
                })
                cool = idx + pos.get("cooldown", 0)
                pos = None
                eq.append(cash)
                continue

            if pos["side"] == "LONG":
                # Full SL on remaining size
                if bar["low"] <= pos["sl"]:
                    ex = _slip(pos["sl"], bar["atr"], bar["close"], "LONG")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], "LONG", fee_bps=MAKER_FEE_BPS)
                    trades.append({
                        "ts": bar.name,
                        "side": "LONG",
                        "entry": round(pos["entry"], 2),
                        "exit": round(ex, 2),
                        "pnl": round(pnl, 4),
                        "result": "SL",
                        "fraction": round(pos["qty_open"] / pos["qty_total"], 4),
                    })
                    cool = idx + pos.get("cooldown", 0)
                    pos = None
                    eq.append(cash)
                    continue

                # TP1 partial: close 50%, move SL to breakeven
                if (not pos["tp1_hit"]) and bar["high"] >= pos["tp1"]:
                    ex = _slip(pos["tp1"], bar["atr"], bar["close"], "LONG")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_tp1"], "LONG", fee_bps=MAKER_FEE_BPS)
                    pos["qty_open"] -= pos["qty_tp1"]
                    pos["tp1_hit"] = True
                    pos["sl"] = pos["entry"]
                    trades.append({
                        "ts": bar.name,
                        "side": "LONG",
                        "entry": round(pos["entry"], 2),
                        "exit": round(ex, 2),
                        "pnl": round(pnl, 4),
                        "result": "TP1",
                        "fraction": round(pos["qty_tp1"] / pos["qty_total"], 4),
                    })

                # TP2 runner exit
                if pos and pos["tp1_hit"] and bar["high"] >= pos["tp2"]:
                    ex = _slip(pos["tp2"], bar["atr"], bar["close"], "LONG")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], "LONG", fee_bps=MAKER_FEE_BPS)
                    trades.append({
                        "ts": bar.name,
                        "side": "LONG",
                        "entry": round(pos["entry"], 2),
                        "exit": round(ex, 2),
                        "pnl": round(pnl, 4),
                        "result": "TP2",
                        "fraction": round(pos["qty_open"] / pos["qty_total"], 4),
                    })
                    cool = idx + pos.get("cooldown", 0)
                    pos = None
                    eq.append(cash)
                    continue

            else:  # SHORT
                if bar["high"] >= pos["sl"]:
                    ex = _slip(pos["sl"], bar["atr"], bar["close"], "SHORT")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], "SHORT", fee_bps=MAKER_FEE_BPS)
                    trades.append({
                        "ts": bar.name,
                        "side": "SHORT",
                        "entry": round(pos["entry"], 2),
                        "exit": round(ex, 2),
                        "pnl": round(pnl, 4),
                        "result": "SL",
                        "fraction": round(pos["qty_open"] / pos["qty_total"], 4),
                    })
                    cool = idx + pos.get("cooldown", 0)
                    pos = None
                    eq.append(cash)
                    continue

                if (not pos["tp1_hit"]) and bar["low"] <= pos["tp1"]:
                    ex = _slip(pos["tp1"], bar["atr"], bar["close"], "SHORT")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_tp1"], "SHORT", fee_bps=MAKER_FEE_BPS)
                    pos["qty_open"] -= pos["qty_tp1"]
                    pos["tp1_hit"] = True
                    pos["sl"] = pos["entry"]
                    trades.append({
                        "ts": bar.name,
                        "side": "SHORT",
                        "entry": round(pos["entry"], 2),
                        "exit": round(ex, 2),
                        "pnl": round(pnl, 4),
                        "result": "TP1",
                        "fraction": round(pos["qty_tp1"] / pos["qty_total"], 4),
                    })

                if pos and pos["tp1_hit"] and bar["low"] <= pos["tp2"]:
                    ex = _slip(pos["tp2"], bar["atr"], bar["close"], "SHORT")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], "SHORT", fee_bps=MAKER_FEE_BPS)
                    trades.append({
                        "ts": bar.name,
                        "side": "SHORT",
                        "entry": round(pos["entry"], 2),
                        "exit": round(ex, 2),
                        "pnl": round(pnl, 4),
                        "result": "TP2",
                        "fraction": round(pos["qty_open"] / pos["qty_total"], 4),
                    })
                    cool = idx + pos.get("cooldown", 0)
                    pos = None
                    eq.append(cash)
                    continue

        # ── Look for new signal when flat ────────────────────────────────────
        if pos is None and idx >= cool:
            w = df.iloc[: i + 1]
            sig = generate_signal(w, state=state, symbol=sym)

            if sig:
                side = _sig(sig, "side")
                ep = _slip(bar["open"], bar["atr"], bar["close"], side)
                sl_p = _sig(sig, "stop_loss_pct", 0.0)
                if sl_p < 0.0005:
                    continue

                sl = ep * (1 - sl_p) if side == "LONG" else ep * (1 + sl_p)
                sl_dist = abs(ep - sl)
                if sl_dist <= 0:
                    continue

                # 1R / 2.5R model
                tp1 = ep + sl_dist if side == "LONG" else ep - sl_dist
                tp2 = ep + (sl_dist * 2.5) if side == "LONG" else ep - (sl_dist * 2.5)

                ema200 = float(df.iloc[i]["ema200"])
                if side == "LONG" and ep < ema200:
                    continue
                if side == "SHORT" and ep > ema200:
                    continue

                risk_amount = equity * RISK_PER_TRADE
                qty = risk_amount / sl_dist
                qty = min(qty, (equity * MAX_NOTIONAL_FRAC) / ep)

                fee = ep * qty * (TAKER_FEE_BPS / 10000)
                cost = qty * ep + fee
                if cost > cash:
                    continue

                pos = {
                    "entry": ep,
                    "qty_total": qty,
                    "qty_open": qty,
                    "qty_tp1": qty * 0.30,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "side": side,
                    "cooldown": _sig(sig, "cooldown_bars", 0),
                    "tp1_hit": False,
                    "bars": 0,
                }
                cash -= cost

        eq.append(cash + (pos["qty_open"] * bar["close"] if pos else 0.0))

    if pos:
        last = df.iloc[-1]
        ex = _slip(last["close"], last["atr"], last["close"], pos["side"])
        cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], pos["side"], fee_bps=MAKER_FEE_BPS)
        trades.append({
            "ts": last.name,
            "side": pos["side"],
            "entry": round(pos["entry"], 2),
            "exit": round(ex, 2),
            "pnl": round(pnl, 4),
            "result": "EOD_CLOSE",
            "fraction": round(pos["qty_open"] / pos["qty_total"], 4),
        })

    pnls = [t["pnl"] for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_los = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins = sum(1 for p in pnls if p > 0)

    eq_arr = np.array(eq if eq else [cap])
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = float(((eq_arr - peak) / peak).min() * 100)

    return {
        "trades": len(trades),
        "win_rate": round(wins / max(len(pnls), 1), 3),
        "profit_factor": round(gross_win / gross_los, 4),
        "final_equity": round(cash, 2),
        "return_pct": round((cash / cap - 1) * 100, 4),
        "max_drawdown_pct": round(dd_pct, 4),
        "avg_rr_realised": round((gross_win / wins) / (gross_los / max(len(pnls) - wins, 1)), 3) if wins > 0 and len(pnls) > wins else 0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    print(json.dumps(run_backtest(a.symbol, a.timeframe, a.start, a.end), indent=2))
