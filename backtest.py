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
RISK_PER_TRADE = 0.01  # 1% of running equity risked per trade
MAX_NOTIONAL_FRAC = 0.25  # hard cap: never more than 25% of equity notional

# Backtest lifecycle defaults; actual values are still mode-aware per signal.
DEFAULT_TP1_R = 1.8
DEFAULT_TP2_R = 4.5
DEFAULT_TP1_QTY_FRAC = 0.20
DEFAULT_MOVE_BE_R = 2.4
MAX_BARS_BY_REGIME = {
    "trend": 72,
    "mean_reversion": 12,
}


def _to_ms(v):
    if not v:
        return None
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _sig(s, k, d=None):
    return getattr(s, k, d) if s is not None else d


def _slip(p, atr, c, side):
    atr_pct = (atr / c) if c else 0.0
    sl = (SLIPPAGE_BPS / 10000) + (atr_pct * SLIPPAGE_ATR_MULT)
    return p * (1 + sl) if side == "LONG" else p * (1 - sl)


def _pnl(entry: float, exit_p: float, qty: float, side: str, fee_bps: float) -> float:
    fee = exit_p * qty * (fee_bps / 10000)
    return (exit_p - entry) * qty - fee if side == "LONG" else (entry - exit_p) * qty - fee


def _close_leg(cash: float, pos: dict, exit_p: float, qty: float, result: str, trades: list):
    pnl = _pnl(pos["entry"], exit_p, qty, pos["side"], MAKER_FEE_BPS)
    cash += pos["entry"] * qty + pnl

    trades.append(
        {
            "ts": pos.get("open_ts", ""),
            "side": pos["side"],
            "entry": round(pos["entry"], 2),
            "exit": round(exit_p, 2),
            "qty": round(qty, 6),
            "pnl": round(pnl, 4),
            "result": result,
        }
    )

    pos["qty_open"] -= qty
    if pos["qty_open"] <= 1e-10:
        return cash, None
    return cash, pos


def fetch_ohlcv_full(sym, tf, since=None, until=None) -> pd.DataFrame:
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
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = compute_indicators(df.reset_index(drop=True))

    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

    return df


def _htf_timeframe_for_symbol(symbol: str, ltf_timeframe: str) -> str:
    # Keep HTF above the backtest LTF so the router can work like live.
    if symbol == "BTC/USDT":
        return "1d" if ltf_timeframe in {"15m", "30m", "1h", "2h", "4h"} else "1h"
    return "4h" if ltf_timeframe in {"15m", "30m", "1h"} else "1d"


def _prepare_signal_levels(sig, entry: float, sl: float):
    sl_dist = abs(entry - sl)

    tp1_pct = _sig(sig, "take_profit_pct", 0.0) or 0.0
    tp2_pct = _sig(sig, "secondary_take_profit_pct", 0.0) or 0.0

    if tp1_pct > 0:
        tp1 = entry * (1 + tp1_pct) if sig.side == "LONG" else entry * (1 - tp1_pct)
    else:
        tp1 = entry + sl_dist * DEFAULT_TP1_R if sig.side == "LONG" else entry - sl_dist * DEFAULT_TP1_R

    if tp2_pct > 0:
        tp2 = entry * (1 + tp2_pct) if sig.side == "LONG" else entry * (1 - tp2_pct)
    else:
        tp2 = entry + sl_dist * DEFAULT_TP2_R if sig.side == "LONG" else entry - sl_dist * DEFAULT_TP2_R

    be_trigger_rr = DEFAULT_MOVE_BE_R
    regime = getattr(sig, "regime", "trend")
    if regime == "mean_reversion":
        be_trigger_rr = 0.6

    be_trigger = entry + sl_dist * be_trigger_rr if sig.side == "LONG" else entry - sl_dist * be_trigger_rr

    tp1_qty_frac = _sig(sig, "tp1_close_fraction", DEFAULT_TP1_QTY_FRAC) or DEFAULT_TP1_QTY_FRAC
    tp2_qty_frac = _sig(sig, "tp2_close_fraction", 1.0 - tp1_qty_frac) or (1.0 - tp1_qty_frac)

    return tp1, tp2, be_trigger, tp1_qty_frac, tp2_qty_frac


def run_backtest(sym, tf, start=None, end=None) -> dict:
    df = fetch_ohlcv_full(sym, tf, _to_ms(start), _to_ms(end))
    if df.empty:
        return {"error": f"no data returned for {sym} on {tf}"}

    htf_tf = _htf_timeframe_for_symbol(sym, tf)
    df_htf = fetch_ohlcv_full(sym, htf_tf, _to_ms(start), _to_ms(end))
    if df_htf.empty:
        return {"error": f"no HTF data returned for {sym} on {htf_tf}"}

    # Fast HTF alignment: one vectorized lookup for the whole run.
    htf_pos = np.searchsorted(df_htf.index.values, df.index.values, side="right") - 1

    cap = 10_000.0
    cash = cap
    pos = None
    trades: list[dict] = []
    eq: list[float] = []
    cool = -1
    state = StrategyState()

    start_idx = max(200, 50)
    for i in range(start_idx, len(df) - 1):
        bar = df.iloc[i + 1]
        idx = i + 1
        bar_atr = float(bar["atr"])
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        equity = cash + (pos["qty_open"] * float(df.iloc[i]["close"]) if pos else 0.0)

        if pos:
            pos["bars"] += 1
            side = pos["side"]

            if pos["bars"] >= pos["max_bars"]:
                ex = _slip(bar_close, bar_atr, bar_close, side)
                cash, pos = _close_leg(cash, pos, ex, pos["qty_open"], "MAX_BARS", trades)
                cool = idx
                eq.append(cash)
                continue

            if side == "LONG":
                sl_hit = bar_low <= pos["sl"]
                tp1_hit = bar_high >= pos["tp1"]
                be_hit = bar_high >= pos["be_trigger"]
                tp2_hit = bar_high >= pos["tp2"]
            else:
                sl_hit = bar_high >= pos["sl"]
                tp1_hit = bar_low <= pos["tp1"]
                be_hit = bar_low <= pos["be_trigger"]
                tp2_hit = bar_low <= pos["tp2"]

            if sl_hit:
                ex = _slip(pos["sl"], bar_atr, bar_close, side)
                cash, pos = _close_leg(cash, pos, ex, pos["qty_open"], "SL", trades)
                cool = idx
                eq.append(cash)
                continue

            if not pos["tp1_hit"] and tp1_hit:
                ex = _slip(pos["tp1"], bar_atr, bar_close, side)
                cash, pos = _close_leg(cash, pos, ex, pos["qty_tp1"], "TP1", trades)
                if pos:
                    pos["tp1_hit"] = True

            if pos and pos["tp1_hit"] and not pos["be_moved"] and be_hit:
                pos["sl"] = pos["entry"]
                pos["be_moved"] = True

            if pos and pos["tp1_hit"] and tp2_hit:
                ex = _slip(pos["tp2"], bar_atr, bar_close, side)
                cash, pos = _close_leg(cash, pos, ex, pos["qty_open"], "TP2", trades)
                cool = idx
                eq.append(cash)
                continue

        if pos is None and idx >= cool:
            w = df.iloc[: i + 1]
            htf_end = htf_pos[idx]
            htf_slice = df_htf.iloc[: htf_end + 1] if htf_end >= 0 else df_htf.iloc[:0]
            sig = generate_signal(w, state=state, symbol=sym, df_htf=htf_slice)

            if sig and sig.side in {"LONG", "SHORT"}:
                ep = _slip(float(bar["open"]), bar_atr, bar_close, sig.side)
                sl_p = _sig(sig, "stop_loss_pct", 0.0)

                if sl_p < 0.0005:
                    eq.append(cash)
                    continue

                sl = ep * (1 - sl_p) if sig.side == "LONG" else ep * (1 + sl_p)
                sl_dist = abs(ep - sl)
                if sl_dist <= 0:
                    eq.append(cash)
                    continue

                tp1, tp2, be_trigger, tp1_frac, tp2_frac = _prepare_signal_levels(sig, ep, sl)
                regime = getattr(sig, "regime", "trend")
                max_bars = MAX_BARS_BY_REGIME.get(regime, 36)

                qty = (equity * RISK_PER_TRADE) / sl_dist
                qty = min(qty, (equity * MAX_NOTIONAL_FRAC) / ep)

                fee = ep * qty * (TAKER_FEE_BPS / 10000)
                cost = qty * ep + fee
                if cost > cash:
                    eq.append(cash)
                    continue

                pos = {
                    "open_ts": str(bar.name),
                    "entry": ep,
                    "qty_open": qty,
                    "qty_tp1": qty * tp1_frac,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "be_trigger": be_trigger,
                    "be_moved": False,
                    "side": sig.side,
                    "cooldown": _sig(sig, "cooldown_bars", 0) or 0,
                    "tp1_hit": False,
                    "bars": 0,
                    "max_bars": max_bars,
                }
                cash -= cost

        eq.append(cash + (pos["qty_open"] * bar_close if pos else 0.0))

    if pos:
        last = df.iloc[-1]
        ex = _slip(float(last["close"]), float(last["atr"]), float(last["close"]), pos["side"])
        cash, pos = _close_leg(cash, pos, ex, pos["qty_open"], "EOD_CLOSE", trades)

    pnls = [t["pnl"] for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_los = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins = sum(1 for p in pnls if p > 0)
    losses = len(pnls) - wins

    eq_arr = np.array(eq if eq else [cap])
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = float(((eq_arr - peak) / peak).min() * 100)

    avg_w = (gross_win / wins) if wins > 0 else 0.0
    avg_l = (gross_los / losses) if losses > 0 else 1e-9
    avg_rr = round(avg_w / avg_l, 3) if avg_l else 0.0

    return {
        "symbol": sym,
        "ltf_timeframe": tf,
        "htf_timeframe": htf_tf,
        "trades": len(trades),
        "win_rate": round(wins / max(len(pnls), 1), 3),
        "profit_factor": round(gross_win / gross_los, 4),
        "final_equity": round(cash, 2),
        "return_pct": round((cash / cap - 1) * 100, 4),
        "max_drawdown_pct": round(dd_pct, 4),
        "avg_rr_realised": avg_rr,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    print(json.dumps(run_backtest(a.symbol, a.timeframe, a.start, a.end), indent=2))
