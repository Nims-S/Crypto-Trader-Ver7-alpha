from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from strategy import compute_indicators


exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
CACHE_DIR = Path(".backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TAKER_FEE_BPS = 6.0
MAKER_FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
SLIPPAGE_ATR_MULT = 0.10
RISK_PER_TRADE = 0.01
MAX_NOTIONAL_FRAC = 0.25


@dataclass
class Position:
    entry: float
    sl: float
    tp1: float
    tp2: float
    qty: float
    qty_tp1: float
    qty_tp2: float
    be_trigger: float
    trail_atr_mult: float
    trail_pct: float
    bars: int = 0
    max_bars: int = 72
    tp1_hit: bool = False
    be_moved: bool = False


def _to_ms(v: Optional[str]):
    if not v:
        return None
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _timeframe_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):
        return int(tf[:-1]) * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def _slip(price: float, atr: float, close: float, side: str) -> float:
    atr_pct = (atr / close) if close else 0.0
    sl = (SLIPPAGE_BPS / 10_000.0) + (atr_pct * SLIPPAGE_ATR_MULT)
    return price * (1 + sl) if side == "LONG" else price * (1 - sl)


def _pnl(entry: float, exit_p: float, qty: float, side: str, fee_bps: float) -> float:
    fee = exit_p * qty * (fee_bps / 10_000.0)
    return (exit_p - entry) * qty - fee if side == "LONG" else (entry - exit_p) * qty - fee


def _cache_path(sym: str, tf: str, since: int | None, until: int | None) -> Path:
    safe = sym.replace("/", "_")
    return CACHE_DIR / f"sanity_{safe}_{tf}_{since or 'none'}_{until or 'none'}.csv"


def fetch_ohlcv_full(sym: str, tf: str, since=None, until=None, use_cache=True) -> pd.DataFrame:
    cache_file = _cache_path(sym, tf, since, until)
    if use_cache and cache_file.exists():
        cached = pd.read_csv(cache_file)
        if cached.empty:
            return pd.DataFrame()
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
        cached = cached.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
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
    df = compute_indicators(df.reset_index(drop=True))
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

    if use_cache:
        df.reset_index().to_csv(cache_file, index=False)
    return df


def get_window(sym: str, tf: str, start=None, end=None, max_bars: int = 0):
    since = _to_ms(start)
    until = _to_ms(end)
    if since is None and max_bars > 0:
        lookback = max(max_bars + 300, 1200)
        since = int(pd.Timestamp.utcnow().timestamp() * 1000) - lookback * _timeframe_to_ms(tf)
    return fetch_ohlcv_full(sym, tf, since, until, use_cache=True)


def signal_btc(df_ltf: pd.DataFrame, df_htf: pd.DataFrame):
    if len(df_ltf) < 50 or len(df_htf) < 20:
        return None

    cur = df_ltf.iloc[-1]
    prev = df_ltf.iloc[-2]
    htf = df_htf.iloc[-1]

    htf_up = htf["close"] > htf["ema200"] and htf["ema20"] >= htf["ema50"]
    ltf_up = cur["ema20"] >= cur["ema50"]
    breakout = cur["close"] > df_ltf["high"].iloc[-21:-1].max()
    reclaim = cur["close"] > cur["ema20"] and prev["close"] <= prev["ema20"]
    momentum = ((cur["close"] - prev["close"]) / prev["close"]) if prev["close"] else 0.0

    if htf_up and ltf_up and (breakout or reclaim or momentum > 0.001):
        entry = float(cur["close"])
        stop = min(float(cur["ema50"]), float(df_ltf["low"].iloc[-20:].min())) * 0.998
        risk = entry - stop
        if risk <= 0:
            return None
        return {
            "entry": entry,
            "stop": stop,
            "tp1": entry + risk * 1.5,
            "tp2": entry + risk * 3.5,
            "be_trigger": entry + risk * 1.0,
            "tp1_frac": 0.20,
            "tp2_frac": 0.80,
            "trail_atr_mult": 1.5,
            "trail_pct": 0.0,
            "max_bars": 72,
        }
    return None


def run_backtest(sym: str, tf: str, start=None, end=None, max_bars: int = 0):
    df = get_window(sym, tf, start=start, end=end, max_bars=max_bars)
    if df.empty:
        return {"error": f"no data returned for {sym} on {tf}"}

    htf_tf = "1d" if tf in {"15m", "30m", "1h", "2h", "4h"} else "4h"
    df_htf = get_window(sym, htf_tf, start=start, end=end, max_bars=max_bars)
    if df_htf.empty:
        return {"error": f"no HTF data returned for {sym} on {htf_tf}"}

    if max_bars > 0:
        warmup = min(300, len(df) - 1)
        df = df.iloc[-(max_bars + warmup):].copy()
        df_htf = df_htf[df_htf.index >= df.index.min()].copy()
        if df_htf.empty:
            return {"error": f"HTF data trimmed away for {sym} on {htf_tf}"}

    cap = 10_000.0
    cash = cap
    pos: Position | None = None
    trades = []
    equity_curve = []
    start_idx = max(50, 20)

    for i in range(start_idx, len(df) - 1):
        bar = df.iloc[i + 1]
        bar_atr = float(bar["atr"])
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if pos:
            pos.bars += 1
            sl_hit = bar_low <= pos.sl
            tp1_hit = bar_high >= pos.tp1
            tp2_hit = bar_high >= pos.tp2
            be_hit = bar_high >= pos.be_trigger

            if sl_hit:
                ex = _slip(pos.sl, bar_atr, bar_close, "LONG")
                pnl = _pnl(pos.entry, ex, pos.qty, "LONG", 2.0)
                cash += pos.entry * pos.qty + pnl
                trades.append((pnl, "SL"))
                pos = None
                equity_curve.append(cash)
                continue

            if not pos.tp1_hit and tp1_hit:
                ex = _slip(pos.tp1, bar_atr, bar_close, "LONG")
                pnl = _pnl(pos.entry, ex, pos.qty_tp1, "LONG", 2.0)
                cash += pos.entry * pos.qty_tp1 + pnl
                pos.qty -= pos.qty_tp1
                pos.tp1_hit = True
                trades.append((pnl, "TP1"))

            if pos.tp1_hit and not pos.be_moved and be_hit:
                pos.sl = pos.entry
                pos.be_moved = True

            if pos.tp1_hit:
                trail = bar_close - (bar_atr * pos.trail_atr_mult)
                pos.sl = max(pos.sl, trail)

            if tp2_hit and pos is not None:
                ex = _slip(pos.tp2, bar_atr, bar_close, "LONG")
                pnl = _pnl(pos.entry, ex, pos.qty, "LONG", 2.0)
                cash += pos.entry * pos.qty + pnl
                trades.append((pnl, "TP2"))
                pos = None
                equity_curve.append(cash)
                continue

        if pos is None:
            sig = signal_btc(df.iloc[: i + 1], df_htf)
            if sig is None:
                equity_curve.append(cash)
                continue

            entry = _slip(float(bar["open"]), bar_atr, bar_close, "LONG")
            sl = float(sig["stop"])
            sl_dist = abs(entry - sl)
            if sl_dist <= 0:
                equity_curve.append(cash)
                continue

            qty = (cash * RISK_PER_TRADE) / sl_dist
            qty = min(qty, (cash * MAX_NOTIONAL_FRAC) / entry)
            if qty <= 0:
                equity_curve.append(cash)
                continue

            pos = Position(
                entry=entry,
                sl=sl,
                tp1=float(sig["tp1"]),
                tp2=float(sig["tp2"]),
                qty=qty,
                qty_tp1=qty * float(sig["tp1_frac"]),
                qty_tp2=qty * float(sig["tp2_frac"]),
                be_trigger=float(sig["be_trigger"]),
                trail_atr_mult=float(sig["trail_atr_mult"]),
                trail_pct=float(sig["trail_pct"]),
                max_bars=int(sig["max_bars"]),
            )
            cash -= entry * qty * (TAKER_FEE_BPS / 10_000.0)

        equity_curve.append(cash + (pos.qty * bar_close if pos else 0.0))

    if pos is not None:
        last = df.iloc[-1]
        ex = _slip(float(last["close"]), float(last["atr"]), float(last["close"]), "LONG")
        pnl = _pnl(pos.entry, ex, pos.qty, "LONG", 2.0)
        cash += pos.entry * pos.qty + pnl
        trades.append((pnl, "EOD"))

    pnls = [p for p, _ in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins = sum(1 for p in pnls if p > 0)

    eq = np.array(equity_curve if equity_curve else [cap])
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)

    return {
        "symbol": sym,
        "ltf_timeframe": tf,
        "htf_timeframe": htf_tf,
        "trades": len(trades),
        "win_rate": round(wins / max(len(trades), 1), 3),
        "profit_factor": round(gross_win / gross_loss, 4),
        "final_equity": round(cash, 2),
        "return_pct": round((cash / cap - 1) * 100, 4),
        "max_drawdown_pct": round(dd, 4),
        "avg_rr_realised": round((gross_win / max(wins, 1)) / (gross_loss / max(len(trades) - wins, 1)), 3) if trades else 0.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-bars", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(run_backtest(a.symbol, a.timeframe, a.start, a.end, max_bars=a.max_bars), indent=2))
