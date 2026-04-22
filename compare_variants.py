from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Optional

import ccxt
import numpy as np
import pandas as pd

from strategy import StrategyState, compute_indicators, generate_signal, get_monday_range

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})

TAKER_FEE_BPS = 6.0
MAKER_FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE = 0.005
MAX_NOTIONAL_FRAC = 0.25
MAX_BARS_IN_TRADE = 24


@dataclass
class VariantResult:
    trades: int
    win_rate: float
    profit_factor: float
    final_equity: float
    return_pct: float
    max_drawdown_pct: float
    avg_rr_realised: float

    def to_dict(self):
        return {
            "trades": self.trades,
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 4),
            "final_equity": round(self.final_equity, 2),
            "return_pct": round(self.return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "avg_rr_realised": round(self.avg_rr_realised, 3),
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


def fetch_ohlcv_full(symbol: str, timeframe: str, since: Optional[int] = None, until: Optional[int] = None) -> pd.DataFrame:
    rows = []
    cur = since
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=1000)
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

    df = compute_indicators(df.reset_index())
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df["ema200"] = df["close"].ewm(span=200, min_periods=1).mean()
    return df


def _close_partial(cash: float, entry: float, exit_price: float, qty: float, side: str) -> tuple[float, float]:
    fee = exit_price * qty * (MAKER_FEE_BPS / 10000)
    pnl = (exit_price - entry) * qty - fee if side == "LONG" else (entry - exit_price) * qty - fee
    cash += qty * exit_price
    return cash, pnl


def _fvg_zone(window: pd.DataFrame, side: str) -> Optional[tuple[float, float]]:
    if len(window) < 3:
        return None
    c2 = window.iloc[-3]
    c1 = window.iloc[-2]
    c0 = window.iloc[-1]

    if side == "LONG":
        fvg_low = float(c1["low"])
        fvg_high = float(c2["high"])
        if fvg_low >= fvg_high:
            return None
        if (fvg_high - fvg_low) < float(c0.get("atr", 0.0)) * 0.2:
            return None
        # Price must actually retrace into the gap.
        if not (float(c0["low"]) <= fvg_high and float(c0["high"]) >= fvg_low):
            return None
        return fvg_high, fvg_low

    fvg_high = float(c1["high"])
    fvg_low = float(c2["low"])
    if fvg_high <= fvg_low:
        return None
    if (fvg_high - fvg_low) < float(c0.get("atr", 0.0)) * 0.2:
        return None
    if not (float(c0["high"]) >= fvg_low and float(c0["low"]) <= fvg_high):
        return None
    return fvg_high, fvg_low


def _fvg_candidate_signal(window: pd.DataFrame, state: StrategyState, symbol: str):
    """Baseline sweep/MSB signal + FVG refinement on the signal bar."""
    base = generate_signal(window, state=state, symbol=symbol)
    if base is None or getattr(base, "side", None) not in {"LONG", "SHORT"}:
        return None

    zone = _fvg_zone(window, base.side)
    if zone is None:
        return None

    fvg_high, fvg_low = zone
    entry = (fvg_high + fvg_low) / 2.0

    if base.side == "LONG":
        stop = fvg_low * 0.999
        if entry <= stop:
            return None
    else:
        stop = fvg_high * 1.001
        if entry >= stop:
            return None

    sl_dist = abs(entry - stop)
    tp1 = entry + sl_dist if base.side == "LONG" else entry - sl_dist
    tp2 = entry + (sl_dist * 2.5) if base.side == "LONG" else entry - (sl_dist * 2.5)

    return {
        "side": base.side,
        "entry": entry,
        "sl": stop,
        "tp1": tp1,
        "tp2": tp2,
        "cooldown": getattr(base, "cooldown_bars", 0) or 0,
    }


def _baseline_signal(window: pd.DataFrame, state: StrategyState, symbol: str):
    sig = generate_signal(window, state=state, symbol=symbol)
    if sig is None or getattr(sig, "side", None) not in {"LONG", "SHORT"}:
        return None
    return {
        "side": sig.side,
        "entry": None,  # baseline uses next bar open
        "sl_pct": float(getattr(sig, "stop_loss_pct", 0.0) or 0.0),
        "tp1_pct": 1.0,  # current execution model TP1 = 1R
        "tp2_mult": 2.5,
        "cooldown": getattr(sig, "cooldown_bars", 0) or 0,
    }


def _simulate(df: pd.DataFrame, symbol: str, mode: str) -> VariantResult:
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

        equity = cash + (pos["qty_open"] * df.iloc[i]["close"] if pos else 0.0)

        if pos:
            pos["bars"] += 1

            if pos["bars"] >= MAX_BARS_IN_TRADE:
                ex = _slip(bar["close"], bar["atr"], bar["close"], pos["side"])
                cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], pos["side"])
                trades.append(pnl)
                cool = idx + pos.get("cooldown", 0)
                pos = None
                eq.append(cash)
                continue

            if pos["side"] == "LONG":
                if bar["low"] <= pos["sl"]:
                    ex = _slip(pos["sl"], bar["atr"], bar["close"], "LONG")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], "LONG")
                    trades.append(pnl)
                    cool = idx + pos.get("cooldown", 0)
                    pos = None
                    eq.append(cash)
                    continue

                if (not pos["tp1_hit"]) and bar["high"] >= pos["tp1"]:
                    ex = _slip(pos["tp1"], bar["atr"], bar["close"], "LONG")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_tp1"], "LONG")
                    pos["qty_open"] -= pos["qty_tp1"]
                    pos["tp1_hit"] = True
                    pos["sl"] = pos["entry"]
                    trades.append(pnl)

                if pos and pos["tp1_hit"] and bar["high"] >= pos["tp2"]:
                    ex = _slip(pos["tp2"], bar["atr"], bar["close"], "LONG")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], "LONG")
                    trades.append(pnl)
                    cool = idx + pos.get("cooldown", 0)
                    pos = None
                    eq.append(cash)
                    continue

            else:
                if bar["high"] >= pos["sl"]:
                    ex = _slip(pos["sl"], bar["atr"], bar["close"], "SHORT")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], "SHORT")
                    trades.append(pnl)
                    cool = idx + pos.get("cooldown", 0)
                    pos = None
                    eq.append(cash)
                    continue

                if (not pos["tp1_hit"]) and bar["low"] <= pos["tp1"]:
                    ex = _slip(pos["tp1"], bar["atr"], bar["close"], "SHORT")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_tp1"], "SHORT")
                    pos["qty_open"] -= pos["qty_tp1"]
                    pos["tp1_hit"] = True
                    pos["sl"] = pos["entry"]
                    trades.append(pnl)

                if pos and pos["tp1_hit"] and bar["low"] <= pos["tp2"]:
                    ex = _slip(pos["tp2"], bar["atr"], bar["close"], "SHORT")
                    cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], "SHORT")
                    trades.append(pnl)
                    cool = idx + pos.get("cooldown", 0)
                    pos = None
                    eq.append(cash)
                    continue

        if pos is None and idx >= cool:
            window = df.iloc[: i + 1]
            if mode == "baseline":
                sig = _baseline_signal(window, state, symbol)
                if sig:
                    side = sig["side"]
                    ep = _slip(bar["open"], bar["atr"], bar["close"], side)
                    sl_p = sig["sl_pct"]
                    if sl_p < 0.0005:
                        continue
                    sl = ep * (1 - sl_p) if side == "LONG" else ep * (1 + sl_p)
                    sl_dist = abs(ep - sl)
                    if sl_dist <= 0:
                        continue
                    tp1 = ep + sl_dist if side == "LONG" else ep - sl_dist
                    tp2 = ep + (sl_dist * 2.5) if side == "LONG" else ep - (sl_dist * 2.5)
                else:
                    tp1 = tp2 = sl = ep = None
            else:
                sig = _fvg_candidate_signal(window, state, symbol)
                if sig:
                    side = sig["side"]
                    ep = sig["entry"]
                    sl = sig["sl"]
                    tp1 = sig["tp1"]
                    tp2 = sig["tp2"]
                else:
                    tp1 = tp2 = sl = ep = None

            if sig:
                ema200 = float(df.iloc[i]["ema200"])
                if side == "LONG" and ep < ema200:
                    continue
                if side == "SHORT" and ep > ema200:
                    continue

                sl_dist = abs(ep - sl)
                if sl_dist <= 0:
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
                    "cooldown": sig.get("cooldown", 0),
                    "tp1_hit": False,
                    "bars": 0,
                }
                cash -= cost

        eq.append(cash + (pos["qty_open"] * bar["close"] if pos else 0.0))

    if pos:
        last = df.iloc[-1]
        ex = _slip(last["close"], last["atr"], last["close"], pos["side"])
        cash, pnl = _close_partial(cash, pos["entry"], ex, pos["qty_open"], pos["side"])
        trades.append(pnl)

    pnls = trades
    gross_win = sum(p for p in pnls if p > 0)
    gross_los = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins = sum(1 for p in pnls if p > 0)

    eq_arr = np.array(eq if eq else [cap])
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = float(((eq_arr - peak) / peak).min() * 100)

    return VariantResult(
        trades=len(trades),
        win_rate=wins / max(len(pnls), 1),
        profit_factor=gross_win / gross_los,
        final_equity=cash,
        return_pct=(cash / cap - 1) * 100,
        max_drawdown_pct=dd_pct,
        avg_rr_realised=(gross_win / wins) / (gross_los / max(len(pnls) - wins, 1)) if wins > 0 and len(pnls) > wins else 0,
    )


def compare(symbol: str, timeframe: str, start: Optional[str], end: Optional[str]):
    df = fetch_ohlcv_full(symbol, timeframe, _to_ms(start), _to_ms(end))
    baseline = _simulate(df, symbol, "baseline")
    fvg = _simulate(df, symbol, "fvg")
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "baseline": baseline.to_dict(),
        "fvg_variant": fvg.to_dict(),
        "delta": {
            "trades": fvg.trades - baseline.trades,
            "win_rate": round(fvg.win_rate - baseline.win_rate, 4),
            "profit_factor": round(fvg.profit_factor - baseline.profit_factor, 4),
            "return_pct": round(fvg.return_pct - baseline.return_pct, 4),
            "max_drawdown_pct": round(fvg.max_drawdown_pct - baseline.max_drawdown_pct, 4),
            "avg_rr_realised": round(fvg.avg_rr_realised - baseline.avg_rr_realised, 4),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Side-by-side compare: current model vs FVG refinement")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-12-31")
    args = ap.parse_args()
    print(json.dumps(compare(args.symbol, args.timeframe, args.start, args.end), indent=2))
