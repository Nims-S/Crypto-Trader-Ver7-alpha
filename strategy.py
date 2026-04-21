from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class Regime(str, Enum):
    TREND = "trend"
    NO_TRADE = "no_trade"
    HIGH_VOL = "high_vol"


@dataclass
class TradeSignal:
    symbol: str
    side: str
    strategy: str
    regime: str
    confidence: float
    reason: str
    stop_loss_pct: float
    take_profit_pct: float
    secondary_take_profit_pct: float
    trail_pct: float
    trail_atr_mult: float = 0.0
    size_multiplier: float = 1.0
    tp1_close_fraction: float = 0.0
    tp2_close_fraction: float = 0.0
    tp3_pct: float = 0.0
    tp3_enabled: bool = False
    tp3_close_fraction: float = 0.0
    cooldown_bars: int = 0


def _cap(value, low, high):
    return max(low, min(high, float(value)))


def _calc_tp(atr_pct, floor, mult, cap):
    return _cap(max(floor, float(atr_pct) * mult), floor, cap)


def _parse_args(*args: Any):
    if len(args) == 1:
        return None, args[0]
    if len(args) == 2:
        return args[0], args[1]
    raise TypeError("generate_signal expects df or (symbol, df)")


def no_trade_signal(symbol, regime, reason="No valid setup"):
    return TradeSignal(symbol, "FLAT", "no_trade", str(regime), 0.0, reason, 0, 0, 0, 0)


# ================= INDICATORS =================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma"] + 1e-9)

    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid

    df["trend_strength"] = (df["ema20"] - df["ema50"]).abs() / df["close"]

    return df.dropna().reset_index(drop=True)


# ================= SIGNAL =================
def _higher_timeframe_ok(df: pd.DataFrame, direction: str) -> bool:
    if len(df) < 120 or "timestamp" not in df.columns:
        return False

    # Use a compact slice to avoid pandas block-consolidation memory spikes.
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    htf_src = df.loc[:, cols].tail(800).copy()

    htf = (
        htf_src.set_index("timestamp")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )

    if len(htf) < 30:
        return False

    htf = compute_indicators(htf)
    if htf.empty:
        return False

    r = htf.iloc[-1]
    if direction == "LONG":
        return r["ema20"] > r["ema50"] > r["ema200"] and r["close"] > r["ema50"] and r["rsi"] >= 48
    if direction == "SHORT":
        return r["ema20"] < r["ema50"] < r["ema200"] and r["close"] < r["ema50"] and r["rsi"] <= 52
    return False


def _ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    required = {"ema20", "ema50", "ema200", "rsi", "atr", "atr_pct", "vol_ratio", "bb_width", "trend_strength"}
    if required.issubset(df.columns):
        return df.copy().reset_index(drop=True)
    return compute_indicators(df)


def _structure_ok_long(df: pd.DataFrame) -> bool:
    if len(df) < 12:
        return False
    r = df.iloc[-1]
    recent_high_10 = df["high"].rolling(10).max().iloc[-2]
    recent_low_10 = df["low"].iloc[-10:-3].min()
    higher_high = r["high"] > recent_high_10
    higher_low = df["low"].iloc[-3] > recent_low_10
    return bool(higher_high and higher_low)



def _structure_ok_short(df: pd.DataFrame) -> bool:
    if len(df) < 12:
        return False
    r = df.iloc[-1]
    recent_low_10 = df["low"].rolling(10).min().iloc[-2]
    recent_high_10 = df["high"].iloc[-10:-3].max()
    lower_low = r["low"] < recent_low_10
    lower_high = df["high"].iloc[-3] < recent_high_10
    return bool(lower_low and lower_high)


def _pullback_depth_ok(row: pd.Series) -> bool:
    pullback_depth = abs(row["close"] - row["ema20"]) / row["close"]
    return 0.002 <= pullback_depth <= 0.015


def _body_strength_ok(row: pd.Series) -> bool:
    body_strength = abs(row["close"] - row["open"]) / (row["high"] - row["low"] + 1e-9)
    return body_strength >= 0.6


def _long_signal(symbol: str, df: pd.DataFrame):
    r = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    if not _structure_ok_long(df):
        return None
    if not _pullback_depth_ok(r):
        return None

    pullback = (
        (prev["close"] < prev["ema20"] or prev2["close"] < prev2["ema20"])
        and min(prev["low"], prev2["low"]) <= r["ema20"] * 1.004
    )

    reclaim = (
        r["close"] > r["open"]
        and r["close"] > max(prev["high"], prev2["high"])
        and r["close"] >= r["high"] - (r["high"] - r["low"]) * 0.35
        and _body_strength_ok(r)
    )

    momentum = (
        48 <= r["rsi"] <= 66
        and r["vol_ratio"] > 1.05
        and 0.006 <= r["atr_pct"] <= 0.025
        and 0.025 <= r["bb_width"] <= 0.09
    )

    if pullback and reclaim and momentum:
        return TradeSignal(
            symbol=symbol,
            side="LONG",
            strategy="trend_pullback_v7",
            regime="trend",
            confidence=0.88,
            reason="strict long setup",
            stop_loss_pct=_calc_tp(r["atr_pct"], 0.006, 1.0, 0.020),
            take_profit_pct=_calc_tp(r["atr_pct"], 0.015, 1.7, 0.032),
            secondary_take_profit_pct=_calc_tp(r["atr_pct"], 0.026, 2.8, 0.055),
            trail_pct=_calc_tp(r["atr_pct"], 0.0075, 1.0, 0.022),
            cooldown_bars=12,
        )
    return None


def _short_signal(symbol: str, df: pd.DataFrame):
    r = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    if not _structure_ok_short(df):
        return None
    if not _pullback_depth_ok(r):
        return None

    pullback = (
        (prev["close"] > prev["ema20"] or prev2["close"] > prev2["ema20"])
        and max(prev["high"], prev2["high"]) >= r["ema20"] * 0.996
    )

    reclaim = (
        r["close"] < r["open"]
        and r["close"] < min(prev["low"], prev2["low"])
        and r["close"] <= r["low"] + (r["high"] - r["low"]) * 0.35
        and _body_strength_ok(r)
    )

    momentum = (
        34 <= r["rsi"] <= 52
        and r["vol_ratio"] > 1.05
        and 0.006 <= r["atr_pct"] <= 0.025
        and 0.025 <= r["bb_width"] <= 0.09
    )

    if pullback and reclaim and momentum:
        return TradeSignal(
            symbol=symbol,
            side="SHORT",
            strategy="trend_pullback_v7",
            regime="trend",
            confidence=0.88,
            reason="strict short setup",
            stop_loss_pct=_calc_tp(r["atr_pct"], 0.006, 1.0, 0.020),
            take_profit_pct=_calc_tp(r["atr_pct"], 0.015, 1.7, 0.032),
            secondary_take_profit_pct=_calc_tp(r["atr_pct"], 0.026, 2.8, 0.055),
            trail_pct=_calc_tp(r["atr_pct"], 0.0075, 1.0, 0.022),
            cooldown_bars=12,
        )
    return None


def generate_signal(*args):
    symbol, df = _parse_args(*args)
    symbol = symbol or "BTC/USDT"

    if df is None or len(df) < 80:
        return None
    if "timestamp" not in df.columns:
        return None

    df = _ensure_indicators(df)
    if df.empty:
        return None

    r = df.iloc[-1]
    if r["atr_pct"] > 0.03 or r["bb_width"] > 0.12:
        return None

    if _higher_timeframe_ok(df, "LONG") and r["ema20"] > r["ema50"] > r["ema200"] and r["close"] > r["ema50"]:
        sig = _long_signal(symbol, df)
        if sig:
            return sig

    if _higher_timeframe_ok(df, "SHORT") and r["ema20"] < r["ema50"] < r["ema200"] and r["close"] < r["ema50"]:
        sig = _short_signal(symbol, df)
        if sig:
            return sig

    return None
