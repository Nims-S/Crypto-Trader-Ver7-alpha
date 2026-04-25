from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from price_action_engine import PriceActionEngine

ENGINE = PriceActionEngine()


@dataclass
class StrategyState:
    trades_this_week: int = 0
    allow_shorts: bool = False
    min_atr_pct: float = 0.0038
    min_adx: float = 22.0


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
    be_trigger_rr: float = 0.0
    max_bars_override: int = 0


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _has_core_indicators(df: pd.DataFrame) -> bool:
    return (
        df is not None
        and not df.empty
        and all(c in df.columns for c in ("atr", "rolling_body", "ema20", "ema50", "ema200", "adx"))
    )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.rolling(14).mean()
    df["rolling_body"] = (df["close"] - df["open"]).abs().rolling(20).mean()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_mask = (up_move > down_move) & (up_move > 0)
    minus_mask = (down_move > up_move) & (down_move > 0)
    plus_dm.loc[plus_mask] = up_move.loc[plus_mask]
    minus_dm.loc[minus_mask] = down_move.loc[minus_mask]

    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df["adx"] = dx.rolling(14).mean()

    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df if _has_core_indicators(df) else compute_indicators(df)


def _swing_low(df, n=20):
    return float(df["low"].iloc[-n:].min())


def _swing_high(df, n=20):
    return float(df["high"].iloc[-n:].max())


def _atr_pct(cur: pd.Series) -> float:
    close = float(cur.get("close", 0.0) or 0.0)
    atr = float(cur.get("atr", 0.0) or 0.0)
    return atr / close if close else 0.0


def _engine_regime(df: pd.DataFrame) -> str:
    """Make the price action engine drive a clean regime check."""
    prepared = ENGINE.get_swing_highs_lows(df)
    return ENGINE.determine_regime(prepared)


def _engine_bullish_trigger(df: pd.DataFrame) -> bool:
    prepared = ENGINE.get_swing_highs_lows(df)
    return ENGINE.check_bullish_trigger(prepared)


# ─────────────────────────────────────────────────────────────
# BTC TREND ENGINE
# ─────────────────────────────────────────────────────────────

def generate_signal_trend_btc(df_ltf, df_htf, symbol, state=None):
    if df_ltf is None or df_htf is None or len(df_ltf) < 80 or len(df_htf) < 80:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    # Engine gates the market first.
    if _engine_regime(df_htf) != "BULL_TREND":
        return None

    # Trigger is the entry confirmation.
    if not _engine_bullish_trigger(df_ltf):
        return None

    cur = df_ltf.iloc[-1]
    prev = df_ltf.iloc[-2]
    htf = df_htf.iloc[-2]
    htf_prev = df_htf.iloc[-3]

    body = abs(float(cur["close"] - cur["open"]))
    body_ok = body >= float(cur["rolling_body"]) * 1.10 if pd.notna(cur["rolling_body"]) else False
    atr_ok = float(cur["atr"]) > float(cur["close"]) * 0.0030 if pd.notna(cur["atr"]) else False
    min_atr_pct = float(getattr(state, "min_atr_pct", 0.0038))
    min_adx = float(getattr(state, "min_adx", 22.0))
    vol_ok = _atr_pct(cur) >= min_atr_pct
    adx_ok = float(cur["adx"]) >= min_adx if pd.notna(cur["adx"]) else False

    htf_up = htf["close"] > htf["ema200"] and htf["ema20"] > htf["ema50"]
    htf_slope_up = htf["ema20"] > htf_prev["ema20"] and htf["ema50"] >= htf_prev["ema50"]
    ltf_up = cur["ema20"] > cur["ema50"]
    trend_ok_long = (htf["ema20"] - htf["ema50"]) / htf["close"] > 0.0030

    pullback = cur["low"] <= cur["ema20"] * 1.005
    reclaim = cur["close"] > cur["ema20"] and prev["close"] <= prev["ema20"]
    prev_highs = df_ltf["high"].iloc[-11:-1]
    breakout = cur["close"] > prev_highs.max()
    entry_condition = (pullback and reclaim) or breakout

    momentum = (cur["close"] - prev["close"]) / prev["close"] if prev["close"] != 0 else 0.0
    momentum_ok = momentum > 0.0025

    runner_mode = adx_ok and float(cur["adx"]) >= 28 and momentum > 0.004 and trend_ok_long

    if htf_up and ltf_up and trend_ok_long and htf_slope_up and body_ok and atr_ok and vol_ok and adx_ok and entry_condition and momentum_ok:
        entry = float(cur["close"])
        recent = df_ltf.iloc[:-1]
        stop = min(_swing_low(recent, 20), float(cur["ema50"])) * 0.998
        risk = entry - stop

        if risk > 0:
            return Signal(
                side="LONG",
                entry_price=entry,
                stop_loss=stop,
                take_profit=entry + risk * (3.5 if runner_mode else 3.0),
                symbol=symbol,
                strategy="btc_trend_v7",
                regime="trend",
                stop_loss_pct=risk / entry,
                take_profit_pct=(risk * (2.5 if runner_mode else 3.0)) / entry,
                secondary_take_profit_pct=(risk * (10.0 if runner_mode else 8.0)) / entry,
                tp1_close_fraction=0.05 if runner_mode else 0.10,
                tp2_close_fraction=0.95 if runner_mode else 0.90,
                be_trigger_rr=2.4 if runner_mode else 1.8,
                max_bars_override=120 if runner_mode else 72,
            )

    return None


# ─────────────────────────────────────────────────────────────
# ALT MEAN REVERSION ENGINE
# ─────────────────────────────────────────────────────────────

def generate_signal_reclaim_alt(df_ltf, symbol, state=None):
    if df_ltf is None or len(df_ltf) < 80:
        return None

    df_ltf = _prepare(df_ltf)
    if df_ltf is None or df_ltf.empty:
        return None

    # Avoid fading strong bearish trends.
    regime = _engine_regime(df_ltf)
    if regime == "BEAR_TREND":
        return None

    cur = df_ltf.iloc[-1]
    prev = df_ltf.iloc[-2]

    lookback = 20
    range_high = df_ltf["high"].iloc[-lookback:].max()
    range_low = df_ltf["low"].iloc[-lookback:].min()

    # Sweep and reclaim long setup.
    sweep_long = cur["low"] < range_low and cur["close"] > range_low
    reclaim_long = cur["close"] > prev["open"] and cur["close"] > prev["close"]
    bullish_trigger = _engine_bullish_trigger(df_ltf)

    if sweep_long and (reclaim_long or bullish_trigger):
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
            strategy="alt_reclaim_v7",
            regime="mean_reversion",
            stop_loss_pct=risk / entry,
            take_profit_pct=(tp1 - entry) / entry,
            secondary_take_profit_pct=(tp2 - entry) / entry,
            tp1_close_fraction=0.6,
            tp2_close_fraction=0.4,
            be_trigger_rr=0.6,
            max_bars_override=12,
        )

    return None


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────

def generate_signal(df, state=None, symbol=None, df_htf=None):
    if symbol == "BTC/USDT":
        return generate_signal_trend_btc(df, df_htf, symbol, state=state)
    return generate_signal_reclaim_alt(df, symbol, state=state)
