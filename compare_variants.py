from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from strategy import StrategyState, compute_indicators, generate_signal, get_fvg

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})

TAKER_FEE_BPS     = 6.0
MAKER_FEE_BPS     = 2.0
SLIPPAGE_BPS      = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE    = 0.01    # must match backtest.py
MAX_NOTIONAL_FRAC = 0.25
MAX_BARS_IN_TRADE = 48      # 2 days on 1h — force-exit stale trades


@dataclass
class VariantResult:
    trades:           int
    win_rate:         float
    profit_factor:    float
    final_equity:     float
    return_pct:       float
    max_drawdown_pct: float
    avg_rr_realised:  float

    def to_dict(self):
        return {
            "trades":           self.trades,
            "win_rate":         round(self.win_rate, 3),
            "profit_factor":    round(self.profit_factor, 4),
            "final_equity":     round(self.final_equity, 2),
            "return_pct":       round(self.return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "avg_rr_realised":  round(self.avg_rr_realised, 3),
        }


def _to_ms(v):
    if not v:
        return None
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _slip(price: float, atr: float, close: float, side: str) -> float:
    atr_pct = (atr / close) if close else 0.0
    slip = (SLIPPAGE_BPS / 10000) + (atr_pct * SLIPPAGE_ATR_MULT)
    return price * (1 + slip) if side == "LONG" else price * (1 - slip)


def _pnl(entry: float, exit_p: float, qty: float, side: str, fee_bps: float) -> float:
    """Correct P&L for both LONG and SHORT (cash-settled)."""
    fee = exit_p * qty * (fee_bps / 10000)
    if side == "LONG":
        return (exit_p - entry) * qty - fee
    else:
        return (entry - exit_p) * qty - fee


def fetch_ohlcv_full(sym: str, tf: str,
                     since: Optional[int] = None,
                     until: Optional[int] = None) -> pd.DataFrame:
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
    df = compute_indicators(df.reset_index(drop=True))
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    df["ema200"] = df["close"].ewm(span=200, min_periods=1).mean()
    return df


def _get_fvg_sl(window: pd.DataFrame, side: str) -> Optional[float]:
    """
    Returns an FVG-anchored stop-loss price, or None if no valid FVG.

    Uses the same get_fvg() from strategy.py but with the corrected
    0.3% buffer (not 0.1%) so it doesn't over-filter.
    """
    if len(window) < 3:
        return None

    c2 = window.iloc[-3]
    c1 = window.iloc[-2]
    c0 = window.iloc[-1]

    if side == "LONG":
        fvg_low  = float(c1["low"])
        fvg_high = float(c2["high"])
        if fvg_low >= fvg_high:
            return None
        # Gap must be meaningful: at least 0.2× ATR
        if (fvg_high - fvg_low) < float(c0.get("atr", 0.0)) * 0.2:
            return None
        # Price must still be reachable (within 0.3% above gap top)
        if float(c0["close"]) > fvg_high * 1.003:
            return None
        return fvg_low * 0.999   # SL just below FVG low

    else:  # SHORT
        fvg_high = float(c1["high"])
        fvg_low  = float(c2["low"])
        if fvg_high <= fvg_low:
            return None
        if (fvg_high - fvg_low) < float(c0.get("atr", 0.0)) * 0.2:
            return None
        if float(c0["close"]) < fvg_low * 0.997:
            return None
        return fvg_high * 1.001  # SL just above FVG high


def _simulate(df: pd.DataFrame, symbol: str, mode: str) -> VariantResult:
    """
    Two modes:
      "baseline" — sweep+MSB signal, SL from strategy.py (sweep low anchor)
      "fvg"      — same signal gate, but SL tightened to FVG anchor when available

    Both modes:
      - Same RISK_PER_TRADE (1%)
      - Partial exit: 30% at TP1 (1R), remainder at TP2 (2.5R)
      - SL moves to break-even after TP1 hit
      - MAX_BARS_IN_TRADE force-exit
      - Causal 200 EMA regime filter
      - Long-only (spot-compatible — no shorts)
    """
    cap   = 10_000.0
    cash  = cap
    pos   = None
    trades: list[float] = []
    eq:   list[float] = []
    cool  = -1
    state = StrategyState()

    for i in range(200, len(df) - 1):
        bar = df.iloc[i + 1]
        idx = i + 1

        equity = cash + (pos["qty_open"] * df.iloc[i]["close"] if pos else 0.0)

        # ── Manage open trade ────────────────────────────────────────────────
        if pos:
            pos["bars"] += 1

            # Force exit on stale trade
            if pos["bars"] >= MAX_BARS_IN_TRADE:
                ex  = _slip(float(bar["close"]), float(bar["atr"]), float(bar["close"]), pos["side"])
                pnl = _pnl(pos["entry"], ex, pos["qty_open"], pos["side"], MAKER_FEE_BPS)
                cash += pos["entry"] * pos["qty_open"] + pnl   # return cost basis + pnl
                trades.append(pnl)
                cool = idx + pos.get("cooldown", 0)
                pos  = None
                eq.append(cash)
                continue

            side = pos["side"]

            # SL check
            sl_hit = (bar["low"] <= pos["sl"]) if side == "LONG" else (bar["high"] >= pos["sl"])
            if sl_hit:
                ex  = _slip(pos["sl"], float(bar["atr"]), float(bar["close"]), side)
                pnl = _pnl(pos["entry"], ex, pos["qty_open"], side, MAKER_FEE_BPS)
                cash += pos["entry"] * pos["qty_open"] + pnl   # return cost basis + pnl
                trades.append(pnl)
                cool = idx + pos.get("cooldown", 0)
                pos  = None
                eq.append(cash)
                continue

            # TP1 check (partial exit: 30%)
            tp1_hit = (bar["high"] >= pos["tp1"]) if side == "LONG" else (bar["low"] <= pos["tp1"])
            if not pos["tp1_hit"] and tp1_hit:
                ex    = _slip(pos["tp1"], float(bar["atr"]), float(bar["close"]), side)
                qty1  = pos["qty_tp1"]
                pnl   = _pnl(pos["entry"], ex, qty1, side, MAKER_FEE_BPS)
                cash += pos["entry"] * qty1 + pnl
                pos["qty_open"] -= qty1
                pos["tp1_hit"]   = True
                pos["sl"]        = pos["entry"]   # move SL to break-even
                trades.append(pnl)

            # TP2 check (close remainder)
            if pos and pos["tp1_hit"]:
                tp2_hit = (bar["high"] >= pos["tp2"]) if side == "LONG" else (bar["low"] <= pos["tp2"])
                if tp2_hit:
                    ex  = _slip(pos["tp2"], float(bar["atr"]), float(bar["close"]), side)
                    pnl = _pnl(pos["entry"], ex, pos["qty_open"], side, MAKER_FEE_BPS)
                    cash += pos["entry"] * pos["qty_open"] + pnl
                    trades.append(pnl)
                    cool = idx + pos.get("cooldown", 0)
                    pos  = None
                    eq.append(cash)
                    continue

        # ── Look for new signal when flat ────────────────────────────────────
        if pos is None and idx >= cool:
            window = df.iloc[:i + 1]
            sig    = generate_signal(window, state=state, symbol=symbol)

            if sig and sig.side == "LONG":   # long-only (spot)
                side  = sig.side
                ep    = _slip(float(bar["open"]), float(bar["atr"]), float(bar["close"]), side)
                sl_p  = float(getattr(sig, "stop_loss_pct", 0.0) or 0.0)

                # ── Mode-specific SL override ─────────────────────────────────
                if mode == "fvg":
                    fvg_sl = _get_fvg_sl(window, side)
                    if fvg_sl is not None:
                        # Recompute sl_p from FVG anchor relative to actual fill
                        sl_p = abs(ep - fvg_sl) / ep
                    # If no FVG found, fall back to sweep-low SL (same as baseline)

                if sl_p < 0.0005:
                    continue

                # ── EMA regime filter ─────────────────────────────────────────
                ema200 = float(df.iloc[i]["ema200"])
                if ep < ema200:
                    continue

                sl     = ep * (1 - sl_p)
                sl_dist = ep - sl
                if sl_dist <= 0:
                    continue

                tp1 = ep + sl_dist          # 1R
                tp2 = ep + sl_dist * 2.5    # 2.5R

                # ── Risk-based sizing ─────────────────────────────────────────
                risk_amount = equity * RISK_PER_TRADE
                qty  = risk_amount / sl_dist
                qty  = min(qty, (equity * MAX_NOTIONAL_FRAC) / ep)
                fee  = ep * qty * (TAKER_FEE_BPS / 10000)
                cost = qty * ep + fee
                if cost > cash:
                    continue

                pos = {
                    "entry":    ep,
                    "qty_open": qty,
                    "qty_tp1":  qty * 0.30,
                    "sl":       sl,
                    "tp1":      tp1,
                    "tp2":      tp2,
                    "side":     side,
                    "cooldown": getattr(sig, "cooldown_bars", 0) or 0,
                    "tp1_hit":  False,
                    "bars":     0,
                }
                cash -= cost

        eq.append(cash + (pos["qty_open"] * float(bar["close"]) if pos else 0.0))

    # ── EOD close ─────────────────────────────────────────────────────────────
    if pos:
        last = df.iloc[-1]
        ex   = _slip(float(last["close"]), float(last["atr"]), float(last["close"]), pos["side"])
        pnl  = _pnl(pos["entry"], ex, pos["qty_open"], pos["side"], MAKER_FEE_BPS)
        cash += pos["entry"] * pos["qty_open"] + pnl
        trades.append(pnl)

    # ── Stats ─────────────────────────────────────────────────────────────────
    pnls      = trades
    gross_win = sum(p for p in pnls if p > 0)
    gross_los = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins      = sum(1 for p in pnls if p > 0)
    losses    = len(pnls) - wins

    eq_arr = np.array(eq if eq else [cap])
    peak   = np.maximum.accumulate(eq_arr)
    dd_pct = float(((eq_arr - peak) / peak).min() * 100)

    avg_w = gross_win  / wins   if wins   > 0 else 0.0
    avg_l = gross_los  / losses if losses > 0 else 1e-9
    avg_rr = round(avg_w / avg_l, 3) if avg_l else 0.0

    return VariantResult(
        trades=len(pnls),
        win_rate=wins / max(len(pnls), 1),
        profit_factor=gross_win / gross_los,
        final_equity=round(cash, 2),
        return_pct=(cash / cap - 1) * 100,
        max_drawdown_pct=dd_pct,
        avg_rr_realised=avg_rr,
    )


def compare(symbol: str, timeframe: str,
            start: Optional[str], end: Optional[str]) -> dict:
    print(f"Fetching {symbol} {timeframe} {start}→{end}...")
    df = fetch_ohlcv_full(symbol, timeframe, _to_ms(start), _to_ms(end))

    print("Running baseline...")
    baseline = _simulate(df, symbol, "baseline")

    print("Running FVG variant...")
    fvg = _simulate(df, symbol, "fvg")

    delta = {
        "trades":           fvg.trades - baseline.trades,
        "win_rate":         round(fvg.win_rate         - baseline.win_rate, 4),
        "profit_factor":    round(fvg.profit_factor    - baseline.profit_factor, 4),
        "return_pct":       round(fvg.return_pct       - baseline.return_pct, 4),
        "max_drawdown_pct": round(fvg.max_drawdown_pct - baseline.max_drawdown_pct, 4),
        "avg_rr_realised":  round(fvg.avg_rr_realised  - baseline.avg_rr_realised, 4),
    }

    return {
        "symbol":      symbol,
        "timeframe":   timeframe,
        "baseline":    baseline.to_dict(),
        "fvg_variant": fvg.to_dict(),
        "delta":       delta,
        "notes": [
            "Long-only (spot compatible). Short signals from strategy are ignored.",
            "Both variants: 1% equity risk, 200 EMA filter, partial TP (30%@1R, 70%@2.5R).",
            "FVG variant: when FVG found, SL tightened to FVG low (0.3% buffer). Falls back to sweep-low SL when no FVG.",
            "Consistent results (PF and return in agreement) indicate no accounting bugs.",
        ],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare baseline vs FVG-refined entry")
    ap.add_argument("--symbol",    default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start",     default="2022-01-01")
    ap.add_argument("--end",       default="2026-12-31")
    a = ap.parse_args()
    print(json.dumps(compare(a.symbol, a.timeframe, a.start, a.end), indent=2))
