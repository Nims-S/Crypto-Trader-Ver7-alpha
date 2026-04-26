from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from strategy import StrategyState, compute_indicators, generate_signal


exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
CACHE_DIR = Path(".backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TAKER_FEE_BPS = 6.0
MAKER_FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE = 0.01
MAX_NOTIONAL_FRAC = 0.25

DEFAULT_TP1_R = 1.8
DEFAULT_TP2_R = 4.5
DEFAULT_TP1_QTY_FRAC = 0.20
DEFAULT_MOVE_BE_R = 1.8
DEFAULT_TRAIL_ATR_MULT = 1.5

MAX_BARS_BY_REGIME = {
    "trend": 30,
    "mean_reversion": 12,
}

REQUIRED_INDICATOR_COLS = {
    "atr",
    "atr_pct",
    "atr_pct_rank",
    "bb_width",
    "bb_width_rank",
    "rolling_body",
    "ema20",
    "ema50",
    "ema200",
    "adx",
    "rsi",
    "macd_hist",
}


@dataclass
class Position:
    side: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    be_trigger: float
    qty_open: float
    qty_tp1: float
    qty_tp2: float
    qty_tp3: float
    bars: int = 0
    max_bars: int = 72
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    be_moved: bool = False
    trail_pct: float = 0.0
    trail_atr_mult: float = DEFAULT_TRAIL_ATR_MULT
    size_multiplier: float = 1.0
    open_ts: str = ""


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


def _close_leg(cash: float, pos: Position, exit_p: float, qty: float, result: str, trades: list):
    if qty <= 0:
        return cash, pos
    qty = min(qty, pos.qty_open)
    pnl = _pnl(pos.entry, exit_p, qty, pos.side, MAKER_FEE_BPS)
    cash += pos.entry * qty + pnl
    trades.append(
        {
            "ts": pos.open_ts,
            "side": pos.side,
            "entry": round(pos.entry, 2),
            "exit": round(exit_p, 2),
            "qty": round(qty, 6),
            "pnl": round(pnl, 4),
            "result": result,
        }
    )
    pos.qty_open -= qty
    return (cash, None) if pos.qty_open <= 1e-10 else (cash, pos)


def _trail_stop(pos: Position, bar_close: float, bar_atr: float):
    if not pos.tp1_hit:
        return

    atr_mult = pos.trail_atr_mult if pos.trail_atr_mult > 0 else DEFAULT_TRAIL_ATR_MULT
    pct = pos.trail_pct if pos.trail_pct > 0 else 0.0

    if pos.side == "LONG":
        trail = bar_close - (bar_atr * atr_mult)
        if pct > 0:
            trail = max(trail, bar_close * (1 - pct))
        pos.sl = max(pos.sl, trail)
    else:
        trail = bar_close + (bar_atr * atr_mult)
        if pct > 0:
            trail = min(trail, bar_close * (1 + pct))
        pos.sl = min(pos.sl, trail)


def _cache_path(sym: str, tf: str, since: int | None, until: int | None) -> Path:
    safe_sym = sym.replace("/", "_")
    since_s = str(since) if since is not None else "none"
    until_s = str(until) if until is not None else "none"
    return CACHE_DIR / f"{safe_sym}_{tf}_{since_s}_{until_s}.csv"


def _normalize_cached_frame(cached: pd.DataFrame) -> pd.DataFrame:
    if cached.empty:
        return cached
    if "timestamp" not in cached.columns:
        return pd.DataFrame()
    cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
    cached = cached.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if not REQUIRED_INDICATOR_COLS.issubset(cached.columns):
        cached = compute_indicators(cached.reset_index())
        if "timestamp" in cached.columns:
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
            cached = cached.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return cached


def fetch_ohlcv_full(sym, tf, since=None, until=None, use_cache=True) -> pd.DataFrame:
    cache_file = _cache_path(sym, tf, since, until)
    if use_cache and cache_file.exists():
        cached = pd.read_csv(cache_file)
        cached = _normalize_cached_frame(cached)
        if not cached.empty:
            return cached

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
    df = compute_indicators(df)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

    if use_cache:
        df.reset_index().to_csv(cache_file, index=False)
    return df


def _htf_timeframe_for_symbol(symbol: str, ltf_timeframe: str) -> str:
    if symbol == "BTC/USDT":
        if ltf_timeframe == "1d":
            return "1w"
        return "1d" if ltf_timeframe in {"15m", "30m", "1h", "2h", "4h"} else "1h"
    return "4h" if ltf_timeframe in {"15m", "30m", "1h"} else "1d"


def _prepare_signal_levels(sig, entry: float, sl: float):
    sl_dist = abs(entry - sl)
    tp1_pct = _sig(sig, "take_profit_pct", 0.0) or 0.0
    tp2_pct = _sig(sig, "secondary_take_profit_pct", 0.0) or 0.0
    tp3_pct = _sig(sig, "tp3_pct", 0.0) or 0.0

    if tp1_pct > 0:
        tp1 = entry * (1 + tp1_pct) if sig.side == "LONG" else entry * (1 - tp1_pct)
    else:
        tp1 = entry + sl_dist * DEFAULT_TP1_R if sig.side == "LONG" else entry - sl_dist * DEFAULT_TP1_R

    if tp2_pct > 0:
        tp2 = entry * (1 + tp2_pct) if sig.side == "LONG" else entry * (1 - tp2_pct)
    else:
        tp2 = entry + sl_dist * DEFAULT_TP2_R if sig.side == "LONG" else entry - sl_dist * DEFAULT_TP2_R

    if tp3_pct > 0:
        tp3 = entry * (1 + tp3_pct) if sig.side == "LONG" else entry * (1 - tp3_pct)
    else:
        tp3 = tp2

    be_trigger_rr = _sig(sig, "be_trigger_rr", DEFAULT_MOVE_BE_R)
    be_trigger = entry + sl_dist * be_trigger_rr if sig.side == "LONG" else entry - sl_dist * be_trigger_rr

    tp1_qty_frac = _sig(sig, "tp1_close_fraction", DEFAULT_TP1_QTY_FRAC) or DEFAULT_TP1_QTY_FRAC
    tp2_qty_frac = _sig(sig, "tp2_close_fraction", 1.0 - tp1_qty_frac) or (1.0 - tp1_qty_frac)
    tp3_qty_frac = _sig(sig, "tp3_close_fraction", 0.0) or 0.0

    return tp1, tp2, tp3, be_trigger, tp1_qty_frac, tp2_qty_frac, tp3_qty_frac


def run_backtest(sym, tf, start=None, end=None, allow_shorts=False, max_bars: int = 0, use_cache: bool = True) -> dict:
    since = _to_ms(start)
    until = _to_ms(end)

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if until:
        until = min(until, now_ms)

    df = fetch_ohlcv_full(sym, tf, since, until, use_cache=use_cache)
    if df.empty:
        return {"error": f"no data returned for {sym} on {tf}"}

    htf_tf = _htf_timeframe_for_symbol(sym, tf)
    df_htf = fetch_ohlcv_full(sym, htf_tf, since, until, use_cache=use_cache)
    if df_htf.empty:
        return {"error": f"no HTF data returned for {sym} on {htf_tf}"}

    if max_bars and max_bars > 0:
        warmup = min(400, len(df) - 1)
        df = df.iloc[-(max_bars + warmup):].copy()
        df_htf = df_htf[df_htf.index >= df.index.min()].copy()
        if df_htf.empty:
            return {"error": f"HTF data trimmed away for {sym} on {htf_tf}"}

    htf_pos = np.searchsorted(df_htf.index.values, df.index.values, side="right") - 1

    cap = 10_000.0
    cash = cap
    pos: Position | None = None
    trades: list[dict] = []
    eq: list[float] = []
    cool = -1
    state = StrategyState(allow_shorts=allow_shorts)
    start_idx = max(260, 50)

    for i in range(start_idx, len(df) - 1):
        bar = df.iloc[i + 1]
        idx = i + 1
        bar_atr = float(bar["atr"])
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if pos:
            pos.bars += 1

            if pos.bars >= pos.max_bars:
                ex = _slip(bar_close, bar_atr, bar_close, pos.side)
                cash, pos = _close_leg(cash, pos, ex, pos.qty_open, "MAX_BARS", trades)
                cool = idx
                eq.append(cash)
                continue

            if pos.side == "LONG":
                sl_hit = bar_low <= pos.sl
                tp1_hit = bar_high >= pos.tp1
                tp2_hit = bar_high >= pos.tp2
                tp3_hit = bar_high >= pos.tp3 if pos.tp3 > 0 else False
                be_hit = bar_high >= pos.be_trigger
            else:
                sl_hit = bar_high >= pos.sl
                tp1_hit = bar_low <= pos.tp1
                tp2_hit = bar_low <= pos.tp2
                tp3_hit = bar_low <= pos.tp3 if pos.tp3 > 0 else False
                be_hit = bar_low <= pos.be_trigger

            if sl_hit:
                ex = _slip(pos.sl, bar_atr, bar_close, pos.side)
                cash, pos = _close_leg(cash, pos, ex, pos.qty_open, "SL", trades)
                cool = idx
                eq.append(cash)
                continue

            if pos and not pos.tp1_hit and tp1_hit:
                ex = _slip(pos.tp1, bar_atr, bar_close, pos.side)
                cash, pos = _close_leg(cash, pos, ex, pos.qty_tp1, "TP1", trades)
                if pos:
                    pos.tp1_hit = True

            if pos and pos.tp1_hit and not pos.be_moved and be_hit:
                pos.sl = pos.entry
                pos.be_moved = True

            if pos and pos.tp1_hit:
                _trail_stop(pos, bar_close, bar_atr)

            if pos and pos.tp3 > 0 and tp3_hit:
                ex = _slip(pos.tp3, bar_atr, bar_close, pos.side)
                cash, pos = _close_leg(cash, pos, ex, pos.qty_open, "TP3", trades)
                cool = idx
                eq.append(cash)
                continue

            if pos and pos.tp1_hit and tp2_hit:
                ex = _slip(pos.tp2, bar_atr, bar_close, pos.side)
                cash, pos = _close_leg(cash, pos, ex, pos.qty_tp2, "TP2", trades)
                if pos is None:
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

                tp1, tp2, tp3, be_trigger, tp1_frac, tp2_frac, tp3_frac = _prepare_signal_levels(sig, ep, sl)
                regime = getattr(sig, "regime", "trend")
                if max_bars and max_bars > 0:
                    max_hold = max_bars
                else:
                    max_hold = _sig(sig, "max_bars_override", 0) or MAX_BARS_BY_REGIME.get(regime, 30)
                size_multiplier = _sig(sig, "size_multiplier", 1.0) or 1.0
                trail_pct = _sig(sig, "trail_pct", 0.0) or 0.0
                trail_atr_mult = _sig(sig, "trail_atr_mult", DEFAULT_TRAIL_ATR_MULT) or DEFAULT_TRAIL_ATR_MULT

                base_qty = (cash * RISK_PER_TRADE) / sl_dist
                qty = min(base_qty * size_multiplier, (cash * MAX_NOTIONAL_FRAC) / ep)
                if qty <= 0:
                    eq.append(cash)
                    continue

                fee = ep * qty * (TAKER_FEE_BPS / 10000)
                cost = qty * ep + fee
                if cost > cash:
                    eq.append(cash)
                    continue

                qty_tp1 = qty * tp1_frac
                qty_tp2 = qty * tp2_frac
                qty_tp3 = qty * tp3_frac if tp3_frac > 0 else max(qty - qty_tp1 - qty_tp2, 0.0)

                if tp3_frac <= 0 and qty_tp1 + qty_tp2 > qty:
                    qty_tp2 = max(qty - qty_tp1, 0.0)
                elif tp3_frac > 0 and qty_tp1 + qty_tp2 + qty_tp3 > qty:
                    qty_tp3 = max(qty - qty_tp1 - qty_tp2, 0.0)

                pos = Position(
                    side=sig.side,
                    entry=ep,
                    sl=sl,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    be_trigger=be_trigger,
                    qty_open=qty,
                    qty_tp1=qty_tp1,
                    qty_tp2=qty_tp2,
                    qty_tp3=qty_tp3,
                    bars=0,
                    max_bars=max_hold,
                    tp1_hit=False,
                    tp2_hit=False,
                    tp3_hit=False,
                    be_moved=False,
                    trail_pct=trail_pct,
                    trail_atr_mult=trail_atr_mult,
                    size_multiplier=size_multiplier,
                    open_ts=str(bar.name),
                )

                cash -= cost

        eq.append(cash + (pos.qty_open * bar_close if pos else 0.0))

    if pos:
        last = df.iloc[-1]
        ex = _slip(float(last["close"]), float(last["atr"]), float(last["close"]), pos.side)
        cash, pos = _close_leg(cash, pos, ex, pos.qty_open, "EOD_CLOSE", trades)

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
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-bars", type=int, default=0, help="Limit the tested window to the most recent N bars for faster iteration")
    ap.add_argument("--no-cache", action="store_true", help="Disable OHLCV caching")
    ap.add_argument("--allow-shorts", action="store_true", help="Enable short trades")
    a = ap.parse_args()
    print(json.dumps(run_backtest(a.symbol, a.timeframe, a.start, a.end, allow_shorts=a.allow_shorts, max_bars=a.max_bars, use_cache=not a.no_cache), indent=2))
