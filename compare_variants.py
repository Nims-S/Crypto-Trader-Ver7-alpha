from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from strategy import StrategyState, compute_indicators, generate_signal

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})

TAKER_FEE_BPS     = 6.0
MAKER_FEE_BPS     = 2.0
SLIPPAGE_BPS      = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE    = 0.01
MAX_NOTIONAL_FRAC = 0.25
MAX_BARS_IN_TRADE = 48


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
    fee = exit_p * qty * (fee_bps / 10000)
    if side == "LONG":
        return (exit_p - entry) * qty - fee
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
    """FVG-anchored SL with 0.3% buffer. Returns None if no valid gap."""
    if len(window) < 3:
        return None
    c2  = window.iloc[-3]
    c1  = window.iloc[-2]
    c0  = window.iloc[-1]
    atr = float(c0.get("atr", 0.0))

    if side == "LONG":
        fvg_low, fvg_high = float(c1["low"]), float(c2["high"])
        if fvg_low >= fvg_high:
            return None
        if (fvg_high - fvg_low) < atr * 0.2:
            return None
        if float(c0["close"]) > fvg_high * 1.003:
            return None
        return fvg_low * 0.999
    else:
        fvg_high, fvg_low = float(c1["high"]), float(c2["low"])
        if fvg_high <= fvg_low:
            return None
        if (fvg_high - fvg_low) < atr * 0.2:
            return None
        if float(c0["close"]) < fvg_low * 0.997:
            return None
        return fvg_high * 1.001


def _simulate(df: pd.DataFrame, symbol: str, cfg: dict) -> VariantResult:
    """
    Universal simulator driven by cfg dict.

    cfg keys:
      use_fvg_sl      bool   — tighten SL to FVG anchor when available
      partial_exit    bool   — 30% at TP1, remainder at TP2
      tp1_r           float  — TP1 as R multiple (only when partial_exit=True)
      tp2_r           float  — full TP (or TP2 when partials on)
      be_after_tp1    bool   — snap SL to BE after TP1 (partial_exit must be True)
      trail_atr_mult  float  — trail SL by N×ATR after TP1; 0 = disabled
      long_only       bool   — ignore SHORT signals
    """
    use_fvg_sl     = cfg.get("use_fvg_sl",     False)
    partial_exit   = cfg.get("partial_exit",   False)
    tp1_r          = cfg.get("tp1_r",          1.0)
    tp2_r          = cfg.get("tp2_r",          2.0)
    be_after_tp1   = cfg.get("be_after_tp1",   True)
    trail_atr_mult = cfg.get("trail_atr_mult", 0.0)
    long_only      = cfg.get("long_only",      True)

    cap   = 10_000.0
    cash  = cap
    pos   = None
    trades: list[float] = []
    eq:   list[float] = []
    cool  = -1
    state = StrategyState()
    # After creating state:
    state = StrategyState()
    # Override min_atr_usd based on symbol
    ATR_FLOORS = {"BTC/USDT": 50.0, "ETH/USDT": 15.0, "SOL/USDT": 2.0}
    state.min_atr_usd = ATR_FLOORS.get(sym, 10.0)
    for i in range(200, len(df) - 1):
        bar       = df.iloc[i + 1]
        idx       = i + 1
        bar_atr   = float(bar["atr"])
        bar_close = float(bar["close"])
        bar_high  = float(bar["high"])
        bar_low   = float(bar["low"])
        bar_open  = float(bar["open"])

        equity = cash + (pos["qty_open"] * float(df.iloc[i]["close"]) if pos else 0.0)

        # ── Manage open trade ────────────────────────────────────────────────
        if pos:
            pos["bars"] += 1
            side = pos["side"]

            # ATR trail (ratchet only) — active after TP1 hit when configured
            if partial_exit and not be_after_tp1 and trail_atr_mult > 0 and pos["tp1_hit"]:
                if side == "LONG":
                    pos["sl"] = max(pos["sl"], bar_close - trail_atr_mult * bar_atr)
                else:
                    pos["sl"] = min(pos["sl"], bar_close + trail_atr_mult * bar_atr)

            # Force-exit stale trade
            if pos["bars"] >= MAX_BARS_IN_TRADE:
                ex  = _slip(bar_close, bar_atr, bar_close, side)
                pnl = _pnl(pos["entry"], ex, pos["qty_open"], side, MAKER_FEE_BPS)
                cash += pos["entry"] * pos["qty_open"] + pnl
                trades.append(pnl)
                cool = idx + pos.get("cooldown", 0)
                pos  = None
                eq.append(cash)
                continue

            # SL hit
            sl_hit = (bar_low <= pos["sl"]) if side == "LONG" else (bar_high >= pos["sl"])
            if sl_hit:
                ex  = _slip(pos["sl"], bar_atr, bar_close, side)
                pnl = _pnl(pos["entry"], ex, pos["qty_open"], side, MAKER_FEE_BPS)
                cash += pos["entry"] * pos["qty_open"] + pnl
                trades.append(pnl)
                cool = idx + pos.get("cooldown", 0)
                pos  = None
                eq.append(cash)
                continue

            if partial_exit:
                # TP1 partial close (30%)
                tp1_confirm = cfg.get("tp1_close_confirm", False)
                if tp1_confirm:
                    tp1_reached = (float(bar["close"]) >= pos["tp1"]) if side == "LONG" else (float(bar["close"]) <= pos["tp1"])
                else:
                    tp1_reached = (bar_high >= pos["tp1"]) if side == "LONG" else (bar_low <= pos["tp1"])
                if not pos["tp1_hit"] and tp1_reached:
                    ex   = _slip(pos["tp1"], bar_atr, bar_close, side)
                    qty1 = pos["qty_tp1"]
                    pnl  = _pnl(pos["entry"], ex, qty1, side, MAKER_FEE_BPS)
                    cash += pos["entry"] * qty1 + pnl
                    pos["qty_open"] -= qty1
                    pos["tp1_hit"]   = True
                    trades.append(pnl)
                    if be_after_tp1:
                        pos["sl"] = pos["entry"]
                    elif trail_atr_mult > 0:
                        if side == "LONG":
                            pos["sl"] = max(pos["sl"], bar_close - trail_atr_mult * bar_atr)
                        else:
                            pos["sl"] = min(pos["sl"], bar_close + trail_atr_mult * bar_atr)

                # TP2 full close (remaining 70%)
                if pos and pos["tp1_hit"]:
                    tp2_reached = (bar_high >= pos["tp2"]) if side == "LONG" else (bar_low <= pos["tp2"])
                    if tp2_reached:
                        ex  = _slip(pos["tp2"], bar_atr, bar_close, side)
                        pnl = _pnl(pos["entry"], ex, pos["qty_open"], side, MAKER_FEE_BPS)
                        cash += pos["entry"] * pos["qty_open"] + pnl
                        trades.append(pnl)
                        cool = idx + pos.get("cooldown", 0)
                        pos  = None
                        eq.append(cash)
                        continue
            else:
                # Single full exit at TP
                tp_reached = (bar_high >= pos["tp"]) if side == "LONG" else (bar_low <= pos["tp"])
                if tp_reached:
                    ex  = _slip(pos["tp"], bar_atr, bar_close, side)
                    pnl = _pnl(pos["entry"], ex, pos["qty_open"], side, MAKER_FEE_BPS)
                    cash += pos["entry"] * pos["qty_open"] + pnl
                    trades.append(pnl)
                    cool = idx + pos.get("cooldown", 0)
                    pos  = None
                    eq.append(cash)
                    continue

        # ── New signal when flat ──────────────────────────────────────────────
        if pos is None and idx >= cool:
            window = df.iloc[:i + 1]
            sig    = generate_signal(window, state=state, symbol=symbol)

            if sig is None or (long_only and sig.side != "LONG"):
                eq.append(cash)
                continue

            side = sig.side
            ep   = _slip(bar_open, bar_atr, bar_close, side)
            sl_p = float(getattr(sig, "stop_loss_pct", 0.0) or 0.0)

            if use_fvg_sl:
                fvg_sl = _get_fvg_sl(window, side)
                if fvg_sl is not None:
                    sl_p = abs(ep - fvg_sl) / ep

            if sl_p < 0.0005:
                eq.append(cash)
                continue

            ema200 = float(df.iloc[i]["ema200"])
            if (side == "LONG" and ep < ema200) or (side == "SHORT" and ep > ema200):
                eq.append(cash)
                continue

            sl      = ep * (1 - sl_p) if side == "LONG" else ep * (1 + sl_p)
            sl_dist = abs(ep - sl)
            if sl_dist <= 0:
                eq.append(cash)
                continue

            qty  = (equity * RISK_PER_TRADE) / sl_dist
            qty  = min(qty, (equity * MAX_NOTIONAL_FRAC) / ep)
            fee  = ep * qty * (TAKER_FEE_BPS / 10000)
            cost = qty * ep + fee
            if cost > cash:
                eq.append(cash)
                continue

            tp_price  = (ep + sl_dist * tp2_r) if side == "LONG" else (ep - sl_dist * tp2_r)
            tp1_price = (ep + sl_dist * tp1_r) if side == "LONG" else (ep - sl_dist * tp1_r)

            pos = {
                "entry":    ep,
                "qty_open": qty,
                "qty_tp1":  qty * 0.30,
                "sl":       sl,
                "tp":       tp_price,
                "tp1":      tp1_price,
                "tp2":      tp_price,
                "side":     side,
                "cooldown": getattr(sig, "cooldown_bars", 0) or 0,
                "tp1_hit":  False,
                "bars":     0,
            }
            cash -= cost

        eq.append(cash + (pos["qty_open"] * bar_close if pos else 0.0))

    # EOD close
    if pos:
        last = df.iloc[-1]
        ex   = _slip(float(last["close"]), float(last["atr"]), float(last["close"]), pos["side"])
        pnl  = _pnl(pos["entry"], ex, pos["qty_open"], pos["side"], MAKER_FEE_BPS)
        cash += pos["entry"] * pos["qty_open"] + pnl
        trades.append(pnl)

    pnls      = trades
    gross_win = sum(p for p in pnls if p > 0)
    gross_los = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins      = sum(1 for p in pnls if p > 0)
    losses    = len(pnls) - wins
    eq_arr    = np.array(eq if eq else [cap])
    peak      = np.maximum.accumulate(eq_arr)
    dd_pct    = float(((eq_arr - peak) / peak).min() * 100)
    avg_w     = gross_win / wins   if wins   > 0 else 0.0
    avg_l     = gross_los / losses if losses > 0 else 1e-9

    return VariantResult(
        trades=len(pnls),
        win_rate=wins / max(len(pnls), 1),
        profit_factor=gross_win / gross_los,
        final_equity=round(cash, 2),
        return_pct=(cash / cap - 1) * 100,
        max_drawdown_pct=dd_pct,
        avg_rr_realised=round(avg_w / avg_l, 3) if avg_l else 0.0,
    )


# ── Variant definitions ───────────────────────────────────────────────────────
# Each isolates one variable so results can be attributed cleanly.

VARIANTS = {
    "A_full_2R": {
        "label":       "Full exit at 2R (backtest.py baseline)",
        "use_fvg_sl":  False, "partial_exit": False,
        "tp2_r":       2.0,   "long_only":    True,
    },
    "B_partial_BE": {
        "label":       "Partial 30%@1R + break-even stop + remainder @2R",
        "use_fvg_sl":  False, "partial_exit": True,
        "tp1_r":       1.0,   "tp2_r":        2.0,
        "be_after_tp1": True, "long_only":    True,
    },
    "C_partial_trail": {
        "label":       "Partial 30%@1R + 1.5×ATR trail + remainder @2.5R",
        "use_fvg_sl":  False, "partial_exit": True,
        "tp1_r":       1.0,   "tp2_r":        2.5,
        "be_after_tp1": False, "trail_atr_mult": 1.5, "long_only": True,
    },
    "D_full_2R_fvg": {
        "label":       "Full exit at 2R + FVG SL tightening",
        "use_fvg_sl":  True,  "partial_exit": False,
        "tp2_r":       2.0,   "long_only":    True,
    },
    "E_trail_fvg": {
        "label":       "Partial trail + FVG SL (combined)",
        "use_fvg_sl":  True,  "partial_exit": True,
        "tp1_r":       1.0,   "tp2_r":        2.5,
        "be_after_tp1": False, "trail_atr_mult": 1.5, "long_only": True,
    },
    "F_trail_tight": {
        "label":        "Partial 30%@1R (close-confirmed) + 1.0×ATR trail + remainder @2.5R",
        "use_fvg_sl":   False, "partial_exit": True,
        "tp1_r":        1.0,   "tp2_r":        2.5,
        "be_after_tp1": False, "trail_atr_mult": 1.0,
        "tp1_close_confirm": True,   # new flag — see simulator change below
        "long_only":    True,
    },
    "G_trail_wider_target": {
        "label":        "Partial 30%@1R + 1.0×ATR trail + remainder @3R",
        "use_fvg_sl":   False, "partial_exit": True,
        "tp1_r":        1.0,   "tp2_r":        3.0,
        "be_after_tp1": False, "trail_atr_mult": 1.0,
        "long_only":    True,
    },
}


def compare(symbol: str, timeframe: str,
            start: Optional[str], end: Optional[str]) -> dict:
    print(f"Fetching {symbol} {timeframe} {start}→{end}...")
    df = fetch_ohlcv_full(symbol, timeframe, _to_ms(start), _to_ms(end))

    results = {}
    for key, cfg in VARIANTS.items():
        label = cfg["label"]
        print(f"  {key}: {label}...")
        cfg_clean = {k: v for k, v in cfg.items() if k != "label"}
        r = _simulate(df, symbol, cfg_clean)
        results[key] = {"label": label, **r.to_dict()}

    best_pf  = max(results, key=lambda k: results[k]["profit_factor"])
    best_ret = max(results, key=lambda k: results[k]["return_pct"])

    return {
        "symbol":       symbol,
        "timeframe":    timeframe,
        "results":      results,
        "best_by_pf":   best_pf,
        "best_by_return": best_ret,
        "what_to_read": {
            "A vs B": "Does break-even partial exit help or hurt?",
            "A vs C": "Does ATR-trail partial exit help or hurt?",
            "A vs D": "Does FVG SL tightening alone help?",
            "C vs E": "Does adding FVG to the best partial exit help?",
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",    default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start",     default="2022-01-01")
    ap.add_argument("--end",       default="2026-12-31")
    a = ap.parse_args()
    print(json.dumps(compare(a.symbol, a.timeframe, a.start, a.end), indent=2))
