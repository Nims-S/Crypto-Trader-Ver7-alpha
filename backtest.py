from __future__ import annotations

import argparse
import json
import time

import ccxt
import numpy as np
import pandas as pd

from strategy import compute_indicators, generate_signal, StrategyState

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})

TAKER_FEE_BPS     = 6.0
MAKER_FEE_BPS     = 2.0
SLIPPAGE_BPS      = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE    = 0.01    # 1% of running equity risked per trade
MAX_NOTIONAL_FRAC = 0.25    # hard cap: never more than 25% of equity notional


def _to_ms(v):
    if not v:
        return None
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _sig(s, k, d=None):
    return getattr(s, k, d) if s is not None else d


def _slip(p, atr, c, side):
    """Apply ATR-scaled slippage to a fill price."""
    atr_pct = (atr / c) if c else 0
    sl = (SLIPPAGE_BPS / 10000) + (atr_pct * SLIPPAGE_ATR_MULT)
    return p * (1 + sl) if side == "LONG" else p * (1 - sl)


def fetch_ohlcv_full(sym, tf, since=None, until=None) -> pd.DataFrame:
    rows = []
    cur  = since
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

    # Precompute ATR + rolling_body ONCE on the full dataset
    df = compute_indicators(df.reset_index(drop=True))

    # Ensure DatetimeIndex for downstream slicing
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

    # ── Causal 200 EMA (no look-ahead) ──────────────────────────────────────
    # ewm with min_periods=1 means early bars get an EMA based only on data
    # available up to that bar — no future information leaks in.
    df["ema200"] = df["close"].ewm(span=200, min_periods=1).mean()

    return df


def run_backtest(sym, tf, start=None, end=None) -> dict:
    df    = fetch_ohlcv_full(sym, tf, _to_ms(start), _to_ms(end))
    cap   = 10_000.0
    cash  = cap
    pos   = None
    trades: list[dict] = []
    eq:    list[float] = []
    cool  = -1
    state = StrategyState()

    # Start at bar 200 so the 200-EMA has warmed up
    for i in range(200, len(df) - 1):
        bar = df.iloc[i + 1]
        idx = i + 1

        # Mark-to-market equity using the CURRENT closed bar's close
        equity = cash + (pos["qty"] * df.iloc[i]["close"] if pos else 0.0)

        # ── Manage open trade ────────────────────────────────────────────────
        if pos:
            hit_sl = bar["low"]  <= pos["sl"]
            hit_tp = bar["high"] >= pos["tp"]

            if hit_sl or hit_tp:
                ex  = pos["sl"] if hit_sl else pos["tp"]
                ex  = _slip(ex, float(bar["atr"]), float(bar["close"]), pos["side"])
                fee = ex * pos["qty"] * (MAKER_FEE_BPS / 10000)
                pnl = (ex - pos["entry"]) * pos["qty"] - fee
                cash += pos["qty"] * ex

                result = "SL" if hit_sl else "TP"
                trades.append({
                    "ts":     str(bar.name),
                    "side":   pos["side"],
                    "entry":  round(pos["entry"], 2),
                    "exit":   round(ex, 2),
                    "pnl":    round(pnl, 4),
                    "result": result,
                })

                # ── Feed realised loss into weekly circuit breaker ────────────
                if result == "SL":
                    state.weekly_pnl += pnl / equity   # fractional loss this week

                cool = idx + pos.get("cooldown", 0)
                pos  = None

        # ── Look for new signal when flat ────────────────────────────────────
        if pos is None and idx >= cool:
            w   = df.iloc[:i + 1]   # window ending at the CLOSED bar (no peek)
            sig = generate_signal(w, state=state, symbol=sym)

            if sig:
                side = _sig(sig, "side")
                ep   = _slip(float(bar["open"]), float(bar["atr"]), float(bar["close"]), side)
                sl_p = _sig(sig, "stop_loss_pct",  0.0)
                tp_p = _sig(sig, "take_profit_pct", 0.0)

                if sl_p < 0.0005:    # degenerate SL < 0.05%, skip
                    continue

                sl = ep * (1 - sl_p)
                tp = ep * (1 + tp_p)

                # ── Causal 200 EMA regime filter ─────────────────────────────
                # Read EMA from bar i (the signal bar), NOT bar i+1 (the fill bar)
                ema200 = float(df.iloc[i]["ema200"])
                if side == "LONG"  and ep < ema200:
                    continue
                if side == "SHORT" and ep > ema200:
                    continue

                # ── Risk-based position sizing ────────────────────────────────
                # Size so that hitting SL costs exactly RISK_PER_TRADE of equity
                risk_amount = equity * RISK_PER_TRADE
                qty = risk_amount / (sl_p * ep)
                qty = min(qty, (equity * MAX_NOTIONAL_FRAC) / ep)   # hard notional cap

                fee  = ep * qty * (TAKER_FEE_BPS / 10000)
                cost = qty * ep + fee
                if cost > cash:
                    continue    # not enough free cash (extremely rare at 1% risk)

                pos = {
                    "entry":    ep,
                    "qty":      qty,
                    "sl":       sl,
                    "tp":       tp,
                    "side":     side,
                    "cooldown": _sig(sig, "cooldown_bars", 0),
                }
                cash -= cost

        # Equity snapshot (open position marked to next bar's close)
        eq.append(cash + (pos["qty"] * float(bar["close"]) if pos else 0.0))

    # ── Close any position still open at end of data ─────────────────────────
    if pos:
        last = df.iloc[-1]
        ex   = _slip(float(last["close"]), float(last["atr"]), float(last["close"]), pos["side"])
        fee  = ex * pos["qty"] * (MAKER_FEE_BPS / 10000)
        pnl  = (ex - pos["entry"]) * pos["qty"] - fee
        cash += pos["qty"] * ex
        trades.append({
            "ts":     str(last.name),
            "side":   pos["side"],
            "entry":  round(pos["entry"], 2),
            "exit":   round(ex, 2),
            "pnl":    round(pnl, 4),
            "result": "EOD_CLOSE",
        })

    # ── Statistics ───────────────────────────────────────────────────────────
    pnls      = [t["pnl"] for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_los = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins      = sum(1 for p in pnls if p > 0)
    losses    = len(pnls) - wins

    eq_arr = np.array(eq if eq else [cap])
    peak   = np.maximum.accumulate(eq_arr)
    dd_pct = float(((eq_arr - peak) / peak).min() * 100)   # true peak-to-trough

    avg_win_rr = (gross_win / wins)      if wins   > 0 else 0.0
    avg_los_rr = (gross_los / losses)    if losses > 0 else 1e-9
    avg_rr     = round(avg_win_rr / avg_los_rr, 3) if avg_los_rr else 0.0

    return {
        "trades":           len(trades),
        "win_rate":         round(wins / max(len(pnls), 1), 3),
        "profit_factor":    round(gross_win / gross_los, 4),
        "final_equity":     round(cash, 2),
        "return_pct":       round((cash / cap - 1) * 100, 4),
        "max_drawdown_pct": round(dd_pct, 4),
        "avg_rr_realised":  avg_rr,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",    default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    print(json.dumps(run_backtest(a.symbol, a.timeframe, a.start, a.end), indent=2))
