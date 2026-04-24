from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class StrategyState:
    trades_this_week: int = 0


@dataclass
class Signal:
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    symbol: str
    strategy: str
    regime: str

    confidence: float = 0.5
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    secondary_take_profit_pct: float = 0.0
    tp1_close_fraction: float = 0.5
    tp2_close_fraction: float = 0.5


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure timestamp index if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    # ATR
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Body size
    df["rolling_body"] = (df["close"] - df["open"]).abs().rolling(20).mean()

    # EMAs
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)

    return df


def _recent_swing_low(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["low"].iloc[-lookback:].min())


def _recent_swing_high(df: pd.DataFrame, lookback: int = 20) -> float:
    return float(df["high"].iloc[-lookback:].max())


# ─────────────────────────────────────────────────────────────
# BTC TREND ENGINE (v2)
# ─────────────────────────────────────────────────────────────

def generate_signal_trend_btc(df_ltf, df_htf, symbol):
    """
    BTC trend engine v2:
    - requires HTF bullish structure
    - requires LTF trend alignment
    - only triggers after pullback + reclaim
    - avoids entries in low-energy chop
    """
    if df_ltf is None or df_htf is None or len(df_ltf) < 80 or len(df_htf) < 80:
        return None

    df_ltf = compute_indicators(df_ltf)
    df_htf = compute_indicators(df_htf)
    if df_ltf.empty or df_htf.empty:
        return None

    cur = df_ltf.iloc[-1]
    prev = df_ltf.iloc[-2]
    htf = df_htf.iloc[-1]

    # ── HTF bias ─────────────────────────────────────────────────────────────
    htf_trend_up = (
        htf["close"] > htf["ema200"]
        and htf["ema20"] > htf["ema50"]
        and htf["ema50"] > htf["ema200"]
    )

    # ── LTF trend + pullback/reclaim ─────────────────────────────────────────
    ltf_trend_up = cur["ema20"] > cur["ema50"] > cur["ema200"]
    pullback_to_ema20 = cur["low"] <= cur["ema20"] * 1.002
    reclaim_above_ema20 = cur["close"] > cur["ema20"] and prev["close"] <= prev["ema20"] * 1.01

    # Displacement / energy filter
    body = abs(float(cur["close"]) - float(cur["open"]))
    body_ok = pd.notna(cur["rolling_body"]) and body >= float(cur["rolling_body"]) * 1.15
    atr_ok = pd.notna(cur["atr"]) and float(cur["atr"]) > float(cur["close"]) * 0.002

    # Avoid taking entries after extended move away from EMA20
    too_extended = float(cur["close"]) > float(cur["ema20"]) * 1.03

    if not (htf_trend_up and ltf_trend_up and pullback_to_ema20 and reclaim_above_ema20 and body_ok and atr_ok):
        return None
    if too_extended:
        return None

    entry = float(cur["close"])
    stop_anchor = min(_recent_swing_low(df_ltf, 20), float(cur["ema50"]))
    stop = stop_anchor * 0.998

    risk = entry - stop
    if risk <= 0:
        return None

    # Wider BTC targets: fewer trades, more room for runners
    tp1 = entry + risk * 1.8
    tp2 = entry + risk * 4.2

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp1,
        symbol=symbol,
        strategy="btc_trend_v2",
        regime="trend",
        confidence=0.72,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp1_close_fraction=0.25,
        tp2_close_fraction=0.75,
    )


# ─────────────────────────────────────────────────────────────
# ALT MEAN REVERSION ENGINE
# ─────────────────────────────────────────────────────────────

def generate_signal_reclaim_alt(df_ltf, symbol):
    if df_ltf is None or len(df_ltf) < 50:
        return None

    df_ltf = compute_indicators(df_ltf)

    cur = df_ltf.iloc[-1]

    lookback = 20
    range_high = df_ltf["high"].iloc[-lookback:].max()
    range_low = df_ltf["low"].iloc[-lookback:].min()

    if cur["low"] < range_low and cur["close"] > range_low:
        entry = float(cur["close"])
        stop = float(cur["low"])
        risk = entry - stop

        if risk <= 0:
            return None

        tp1 = entry + risk * 0.8
        tp2 = entry + risk * 1.4

        return Signal(
            side="LONG",
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp1,
            symbol=symbol,
            strategy="alt_reclaim_v1",
            regime="mean_reversion",
            stop_loss_pct=risk / entry,
            take_profit_pct=(tp1 - entry) / entry,
            secondary_take_profit_pct=(tp2 - entry) / entry,
            tp1_close_fraction=0.6,
            tp2_close_fraction=0.4,
        )

    return None


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────

def generate_signal(df, state=None, symbol=None, df_htf=None):
    if symbol == "BTC/USDT":
        return generate_signal_trend_btc(df, df_htf, symbol)
    return generate_signal_reclaim_alt(df, symbol)
