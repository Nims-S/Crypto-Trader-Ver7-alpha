import pandas as pd
import numpy as np

# =========================
# INDICATORS
# =========================
def compute_indicators(df):
    df = df.copy()

    # EMA trend
    df["ema_fast"] = df["close"].ewm(span=20).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df["atr"] = pd.Series(tr).rolling(14).mean()

    # Bollinger Bands
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()

    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]

    # Volume expansion
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma"] + 1e-9)

    return df


# =========================
# SIGNAL (V7 CORE EDGE)
# =========================
def generate_signal(df):
    """
    Returns:
    dict or None
    """

    if len(df) < 50:
        return None

    r = df.iloc[-1]
    prev = df.iloc[-2]

    # =========================
    # CONDITIONS
    # =========================

    # 1. Volatility squeeze (compression)
    squeeze = r["bb_width"] < 0.05

    # 2. Breakout (closed candle only)
    breakout_long = (
        r["close"] > prev["high"]
        and r["close"] > r["bb_upper"]
    )

    breakout_short = (
        r["close"] < prev["low"]
        and r["close"] < r["bb_lower"]
    )

    # 3. Volume confirmation
    vol_ok = r["vol_ratio"] > 1.5

    # 4. Momentum filter
    long_momentum = r["rsi"] > 55
    short_momentum = r["rsi"] < 45

    # =========================
    # LONG SIGNAL
    # =========================
    if squeeze and breakout_long and vol_ok and long_momentum:

        entry = r["close"]
        sl = entry - 1.5 * r["atr"]
        tp = entry + 3 * r["atr"]

        return {
            "side": "long",
            "entry": entry,
            "sl": sl,
            "tp": tp
        }

    # =========================
    # SHORT SIGNAL
    # =========================
    if squeeze and breakout_short and vol_ok and short_momentum:

        entry = r["close"]
        sl = entry + 1.5 * r["atr"]
        tp = entry - 3 * r["atr"]

        return {
            "side": "short",
            "entry": entry,
            "sl": sl,
            "tp": tp
        }

    return None
