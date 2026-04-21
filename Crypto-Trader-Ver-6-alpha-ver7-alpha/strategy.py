from dataclasses import dataclass
from enum import Enum

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


def _cap(value, low, high):
    return max(low, min(high, float(value)))


def _calc_tp(atr_pct, floor, mult, cap):
    return _cap(max(floor, float(atr_pct) * mult), floor, cap)


def _volatility_size_multiplier(atr_pct, base, floor, ceiling):
    atr_pct = max(1e-6, float(atr_pct))
    vol_scale = _cap(0.018 / atr_pct, floor, ceiling)
    return _cap(base * vol_scale, 0.5, 1.15)


def no_trade_signal(symbol, regime, reason="No valid setup"):
    return TradeSignal(
        symbol=symbol,
        side="FLAT",
        strategy="no_trade",
        regime=str(regime),
        confidence=0.0,
        reason=reason,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        secondary_take_profit_pct=0.0,
        trail_pct=0.0,
        size_multiplier=0.0,
        tp1_close_fraction=0.0,
        tp2_close_fraction=0.0,
        tp3_pct=0.0,
        tp3_enabled=False,
        tp3_close_fraction=0.0,
    )


# ================= INDICATORS =================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().replace(0, pd.NA)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    df["vol_avg"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"]

    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid

    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    df["trend_strength"] = (df["ema20"] - df["ema50"]).abs() / df["close"]

    return df.dropna()


# ================= REGIME =================
def detect_regime(row):
    if row["atr_pct"] > 0.028 or row["bb_width"] > 0.12:
        return Regime.HIGH_VOL
    if row["ema20"] > row["ema50"] > row["ema200"]:
        return Regime.TREND
    return Regime.NO_TRADE


# ================= SIGNAL =================
def generate_signal(symbol: str, df: pd.DataFrame):
    if df.empty or len(df) < 60:
        return no_trade_signal(symbol, Regime.NO_TRADE, "Insufficient data")

    row = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    regime = detect_regime(row)

    if regime != Regime.TREND:
        return no_trade_signal(symbol, regime, "Trend structure not present")

    # Single high-conviction pattern:
    # 1) established uptrend
    # 2) shallow pullback into EMA20 over the last 2-3 candles
    # 3) reclaim candle closes back above EMA20 with strong volume and bullish close
    trend_ok = (
        row["ema20"] > row["ema50"] > row["ema200"]
        and row["close"] > row["ema50"]
        and row["ema50"] > row["ema200"]
    )

    if not trend_ok:
        return no_trade_signal(symbol, regime, "Trend stack failed")

    recent_pullback = (
        min(df.iloc[-3:]["low"].min(), prev["low"], prev2["low"]) <= row["ema20"] * 1.002
        and (prev["close"] < prev["ema20"] or prev2["close"] < prev2["ema20"])
    )

    reclaim_candle = (
        row["close"] > row["open"]
        and row["close"] > row["ema20"]
        and row["close"] > prev["high"]
        and row["close"] >= row["high"] - (row["high"] - row["low"]) * 0.35
    )

    momentum_ok = (
        48 <= row["rsi"] <= 64
        and row["vol_ratio"] >= 1.20
        and 0.007 <= row["atr_pct"] <= 0.022
        and 0.025 <= row["bb_width"] <= 0.085
        and row["close"] <= row["ema20"] * 1.012
    )

    if not (recent_pullback and reclaim_candle and momentum_ok):
        return no_trade_signal(symbol, regime, "Pullback continuation filter failed")

    atr = float(row["atr_pct"])
    trail_atr = 1.15 if symbol == "BTC/USDT" else 1.2 if symbol == "ETH/USDT" else 1.25 if symbol == "SOL/USDT" else 1.15

    return TradeSignal(
        symbol=symbol,
        side="LONG",
        strategy="trend_pullback_v7",
        regime=regime.value,
        confidence=0.90,
        reason="Strict pullback continuation",
        stop_loss_pct=_calc_tp(atr, 0.006, 1.0, 0.018),
        take_profit_pct=_calc_tp(atr, 0.016, 1.8, 0.030),
        secondary_take_profit_pct=_calc_tp(atr, 0.028, 3.0, 0.050),
        trail_pct=_calc_tp(atr, 0.008, 1.0, 0.020),
        trail_atr_mult=trail_atr,
        size_multiplier=_volatility_size_multiplier(atr, 0.95, 0.65, 1.05),
        tp1_close_fraction=0.45,
        tp2_close_fraction=0.35,
        tp3_pct=_calc_tp(atr, 0.045, 4.5, 0.080),
        tp3_enabled=True,
        tp3_close_fraction=0.20,
    )
