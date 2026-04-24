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

# ── Exit structure (Variant C — confirmed best across 7 variants) ─────────────
TP1_R          = 1.5    # first target: 1R — take partial profit here
TP2_R          = 3    # second target: 2.5R — close remainder here
TP1_QTY_FRAC   = 0.30   # fraction closed at TP1
MOVE_BE_R      = 1.8   # NEW)
MAX_BARS       = 36     # force-exit any trade open longer than 48 bars (2 days on 1h)


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


def _pnl(entry: float, exit_p: float, qty: float, side: str, fee_bps: float) -> float:
    """P&L for a closed leg, net of fees. Correct for both LONG and SHORT."""
    fee = exit_p * qty * (fee_bps / 10000)
    return (exit_p - entry) * qty - fee if side == "LONG" else (entry - exit_p) * qty - fee


def _close_leg(cash: float, pos: dict, exit_p: float, qty: float,
               result: str, trades: list, cool_idx: int) -> tuple[float, dict | None]:
    """
    Close `qty` units of `pos` at `exit_p`.
    Updates cash, appends a trade record, returns (new_cash, pos_or_None).
    pos is set to None only when qty equals the full remaining open quantity.
    """
    pnl = _pnl(pos["entry"], exit_p, qty, pos["side"], MAKER_FEE_BPS)
    cash += pos["entry"] * qty + pnl   # return cost basis + profit/loss

    trades.append({
        "ts":     pos.get("open_ts", ""),
        "side":   pos["side"],
        "entry":  round(pos["entry"], 2),
        "exit":   round(exit_p, 2),
        "qty":    round(qty, 6),
        "pnl":    round(pnl, 4),
        "result": result,
    })

    pos["qty_open"] -= qty
    if pos["qty_open"] <= 1e-10:
        return cash, None
    return cash, pos


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

    # Causal 200 EMA — ewm min_periods=1 ensures no future data leaks
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

    for i in range(200, len(df) - 1):
        bar       = df.iloc[i + 1]
        idx       = i + 1
        bar_atr   = float(bar["atr"])
        bar_close = float(bar["close"])
        bar_high  = float(bar["high"])
        bar_low   = float(bar["low"])

        # Mark-to-market equity on the CLOSED bar (bar i, not the fill bar i+1)
        equity = cash + (pos["qty_open"] * float(df.iloc[i]["close"]) if pos else 0.0)

        # ── Manage open trade ─────────────────────────────────────────────────
        if pos:
            pos["bars"] += 1
            side = pos["side"]

           

            # ── (5) Force-exit stale trade ────────────────────────────────────
            # Fires BEFORE SL/TP checks so a trade that has been dragging for
            # MAX_BARS without resolution gets closed at market regardless.
            if pos["bars"] >= MAX_BARS:
                ex = _slip(bar_close, bar_atr, bar_close, side)
                cash, pos = _close_leg(cash, pos, ex, pos["qty_open"],
                                       "MAX_BARS", trades, idx)
                cool = idx
                eq.append(cash)
                continue

            # ── (1) SL hit — close entire remaining position ──────────────────
            sl_hit = bar_low <= pos["sl"]   # LONG only
            if sl_hit:
                ex = _slip(pos["sl"], bar_atr, bar_close, side)
                cash, pos = _close_leg(cash, pos, ex, pos["qty_open"],
                                       "SL", trades, idx)
                # Feed fractional loss into weekly circuit breaker
                sl_pnl = trades[-1]["pnl"]
                state.weekly_pnl += sl_pnl / equity
                cool = idx
                pos  = None
                eq.append(cash)
                continue

            # ── (2) TP1 hit — close 30%, begin trailing ───────────────────────
            # Triggered on wick (bar["high"] >= tp1). Close-confirmation
            # was tested as variant F and was *worse* than wick-trigger,
            # so we keep the wick trigger here.
            if not pos["tp1_hit"] and bar_high >= pos["tp1"]:
                ex = _slip(pos["tp1"], bar_atr, bar_close, side)
                cash, pos = _close_leg(cash, pos, ex, pos["qty_tp1"],
                                       "TP1", trades, idx)
                if pos:   # trade still open with remaining 70%
                    pos["tp1_hit"] = True
                    # Initialise trail from current bar
                    pos["sl"] = max(pos["sl"], bar_close - TRAIL_ATR_MULT * bar_atr)

            # ── MOVE SL TO BREAKEVEN ─────────────────────────────────
            if pos and not pos["be_moved"] and bar_high >= pos["be_trigger"]:
                pos["sl"] = pos["entry"]
                pos["be_moved"] = True

            # ── (4) TP2 hit — close remainder ─────────────────────────────────
            # Only checked after TP1 has been hit.
            if pos and pos["tp1_hit"] and bar_high >= pos["tp2"]:
                ex = _slip(pos["tp2"], bar_atr, bar_close, side)
                cash, pos = _close_leg(cash, pos, ex, pos["qty_open"],
                                       "TP2", trades, idx)
                cool = idx
                pos  = None
                eq.append(cash)
                continue

        # ── Look for new signal when flat ──────────────────────────────────────
        if pos is None and idx >= cool:
            w   = df.iloc[:i + 1]   # window ending at CLOSED bar — no peek
            sig = generate_signal(w, state=state, symbol=sym)

            if sig and sig.side == "LONG":   # long-only (spot compatible)
                ep   = _slip(float(bar["open"]), bar_atr, bar_close, "LONG")
                sl_p = _sig(sig, "stop_loss_pct", 0.0)

                if sl_p < 0.0005:   # degenerate SL, skip
                    eq.append(cash)
                    continue

                sl = ep * (1 - sl_p)

                # Causal 200 EMA regime filter — read from signal bar i, not fill bar
                ema200 = float(df.iloc[i]["ema200"])
                if ep < ema200:
                    eq.append(cash)
                    continue

                sl_dist = ep - sl
                if sl_dist <= 0:
                    eq.append(cash)
                    continue

                # Risk-based sizing: SL hit costs exactly 1% of equity
                qty  = (equity * RISK_PER_TRADE) / sl_dist
                qty  = min(qty, (equity * MAX_NOTIONAL_FRAC) / ep)

                fee  = ep * qty * (TAKER_FEE_BPS / 10000)
                cost = qty * ep + fee
                if cost > cash:
                    eq.append(cash)
                    continue

                pos = {
                    "open_ts":  str(bar.name),
                    "entry":    ep,
                    "qty_open": qty,              # decrements as legs close
                    "qty_tp1":  qty * TP1_QTY_FRAC,
                    "sl":       sl,
                    "tp1":      ep + sl_dist * TP1_R,
                    "tp2":      ep + sl_dist * TP2_R,
                    "be_trigger": ep + sl_dist * MOVE_BE_R,   # NEW
                    "be_moved":   False,                      # NEW
                    "side":     "LONG",
                    "cooldown": _sig(sig, "cooldown_bars", 0) or 0,
                    "tp1_hit":  False,
                    "bars":     0,
                }
                cash -= cost

        eq.append(cash + (pos["qty_open"] * bar_close if pos else 0.0))

    # ── Close any position still open at end of data ──────────────────────────
    if pos:
        last = df.iloc[-1]
        ex   = _slip(float(last["close"]), float(last["atr"]),
                     float(last["close"]), pos["side"])
        cash, pos = _close_leg(cash, pos, ex, pos["qty_open"],
                               "EOD_CLOSE", trades, len(df))

    # ── Statistics ────────────────────────────────────────────────────────────
    pnls      = [t["pnl"] for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_los = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins      = sum(1 for p in pnls if p > 0)
    losses    = len(pnls) - wins

    eq_arr = np.array(eq if eq else [cap])
    peak   = np.maximum.accumulate(eq_arr)
    dd_pct = float(((eq_arr - peak) / peak).min() * 100)

    avg_w  = (gross_win / wins)   if wins   > 0 else 0.0
    avg_l  = (gross_los / losses) if losses > 0 else 1e-9
    avg_rr = round(avg_w / avg_l, 3) if avg_l else 0.0

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
