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


def add_indicators(df: pd.DataFrame):
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    return df


# ─────────────────────────────────────────────────────────────
# BTC TREND ENGINE
# ─────────────────────────────────────────────────────────────

def generate_signal_trend_btc(df_ltf, df_htf, symbol):
    if df_ltf is None or df_htf is None or len(df_ltf) < 50 or len(df_htf) < 50:
        return None

    df_ltf = add_indicators(df_ltf)
    df_htf = add_indicators(df_htf)

    cur = df_ltf.iloc[-1]

    # HTF bias
    htf_trend_up = df_htf.iloc[-1]["ema20"] > df_htf.iloc[-1]["ema50"]

    # LTF continuation
    trend_up = cur["ema20"] > cur["ema50"]

    if not (htf_trend_up and trend_up):
        return None

    entry = float(cur["close"])
    stop = float(df_ltf["low"].iloc[-5:].min())

    risk = entry - stop
    if risk <= 0:
        return None

    tp1 = entry + risk * 1.5
    tp2 = entry + risk * 3.5

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp1,
        symbol=symbol,
        strategy="btc_trend_v1",
        regime="trend",
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp1_close_fraction=0.3,
        tp2_close_fraction=0.7,
    )


# ─────────────────────────────────────────────────────────────
# ALT MEAN REVERSION ENGINE
# ─────────────────────────────────────────────────────────────

def generate_signal_reclaim_alt(df_ltf, symbol):
    if df_ltf is None or len(df_ltf) < 50:
        return None

    cur = df_ltf.iloc[-1]

    lookback = 20
    range_high = df_ltf["high"].iloc[-lookback:].max()
    range_low = df_ltf["low"].iloc[-lookback:].min()

    # sweep below range
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
    else:
        return generate_signal_reclaim_alt(df, symbol)


# compatibility

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return df
