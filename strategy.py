from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from price_action_engine import PriceActionEngine

ENGINE = PriceActionEngine()


@dataclass
class StrategyState:
    trades_this_week: int = 0
    allow_shorts: bool = False
    min_atr_pct: float = 0.0032
    min_adx: float = 18.0


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
    tp3_pct: float = 0.0
    tp3_close_fraction: float = 0.0
    trail_pct: float = 0.0
    trail_atr_mult: float = 0.0
    tp1_close_fraction: float = 0.5
    tp2_close_fraction: float = 0.5
    be_trigger_rr: float = 0.0
    max_bars_override: int = 0
    cooldown_bars: int = 0
    size_multiplier: float = 1.0


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


def _atr_pct(cur: pd.Series) -> float:
    close = float(cur.get("close", 0.0) or 0.0)
    atr = float(cur.get("atr", 0.0) or 0.0)
    return atr / close if close else 0.0


def _prepare_engine_frame(df: pd.DataFrame) -> pd.DataFrame:
    return ENGINE.prepare(df)


def _engine_regime_marked(df_marked: pd.DataFrame) -> str:
    return ENGINE.determine_regime(df_marked)


def _engine_bullish_trigger_marked(df_marked: pd.DataFrame) -> bool:
    return ENGINE.check_bullish_trigger(df_marked)


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

    df_ltf_eng = _prepare_engine_frame(df_ltf)
    df_htf_eng = _prepare_engine_frame(df_htf)

    htf_regime = _engine_regime_marked(df_htf_eng)
    if htf_regime == "BEAR_TREND":
        return None

    cur = df_ltf.iloc[-1]
    prev = df_ltf.iloc[-2]
    htf = df_htf.iloc[-2]
    htf_prev = df_htf.iloc[-3]

    body = abs(float(cur["close"] - cur["open"]))
    body_ok = body >= float(cur["rolling_body"]) * 0.80 if pd.notna(cur["rolling_body"]) else True
    atr_ok = float(cur["atr"]) > float(cur["close"]) * 0.0020 if pd.notna(cur["atr"]) else True
    min_atr_pct = float(getattr(state, "min_atr_pct", 0.0032))
    min_adx = float(getattr(state, "min_adx", 18.0))
    vol_ok = _atr_pct(cur) >= min_atr_pct
    adx_ok = float(cur["adx"]) >= min_adx if pd.notna(cur["adx"]) else True

    htf_up = htf["close"] > htf["ema200"] and htf["ema20"] > htf["ema50"]
    htf_slope_up = htf["ema20"] >= htf_prev["ema20"]
    ltf_up = cur["ema20"] >= cur["ema50"]
    trend_ok_long = (htf["ema20"] - htf["ema50"]) / htf["close"] > 0.0015

    pullback = cur["low"] <= cur["ema20"] * 1.015
    reclaim = cur["close"] > cur["ema20"] and prev["close"] <= prev["ema20"]
    prev_highs = df_ltf["high"].iloc[-11:-1]
    breakout = cur["close"] > prev_highs.max()
    momentum = (cur["close"] - prev["close"]) / prev["close"] if prev["close"] != 0 else 0.0
    momentum_ok = momentum > 0.0008

    bullish_trigger = _engine_bullish_trigger_marked(df_ltf_eng)
    regime_bias_ok = htf_regime in {"BULL_TREND", "RANGING", "UNKNOWN"}
    strong_entry_ok = breakout or (pullback and reclaim) or bullish_trigger
    fallback_entry_ok = htf_up and ltf_up and (breakout or pullback or bullish_trigger)
    runner_mode = adx_ok and float(cur["adx"]) >= 24 and momentum > 0.0030 and trend_ok_long

    if htf_up and ltf_up and trend_ok_long and htf_slope_up and vol_ok and regime_bias_ok and momentum_ok and (strong_entry_ok or fallback_entry_ok):
        if not (body_ok and atr_ok and adx_ok):
            confidence = 0.35
            tp_mult = 2.2
            tp2_mult = 4.5
            tp3_mult = 6.5
            tp1_frac = 0.15
            tp2_frac = 0.85
            be_rr = 1.3
            max_bars = 60
            size_mult = 0.8
        else:
            confidence = 0.5
            tp_mult = 3.5 if runner_mode else 3.0
            tp2_mult = 10.0 if runner_mode else 8.0
            tp3_mult = 12.0 if runner_mode else 9.0
            tp1_frac = 0.05 if runner_mode else 0.10
            tp2_frac = 0.95 if runner_mode else 0.90
            be_rr = 2.4 if runner_mode else 1.8
            max_bars = 120 if runner_mode else 72
            size_mult = 1.0

        entry = float(cur["close"])
        recent = df_ltf.iloc[:-1]
        stop = min(_swing_low(recent, 20), float(cur["ema50"])) * 0.998
        risk = entry - stop

        if risk > 0:
            return Signal(
                side="LONG",
                entry_price=entry,
                stop_loss=stop,
                take_profit=entry + risk * tp_mult,
                symbol=symbol,
                strategy="btc_trend_v9",
                regime="trend",
                confidence=confidence,
                stop_loss_pct=risk / entry,
                take_profit_pct=(risk * tp_mult) / entry,
                secondary_take_profit_pct=(risk * tp2_mult) / entry,
                tp3_pct=(risk * tp3_mult) / entry,
                tp3_close_fraction=0.0,
                trail_pct=0.0,
                trail_atr_mult=0.0,
                tp1_close_fraction=tp1_frac,
                tp2_close_fraction=tp2_frac,
                be_trigger_rr=be_rr,
                max_bars_override=max_bars,
                cooldown_bars=0,
                size_multiplier=size_mult,
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

    df_ltf_eng = _prepare_engine_frame(df_ltf)

    regime = _engine_regime_marked(df_ltf_eng)
    if regime == "BEAR_TREND":
        return None

    cur = df_ltf.iloc[-1]
    prev = df_ltf.iloc[-2]

    lookback = 20
    range_low = df_ltf["low"].iloc[-lookback:].min()

    sweep_long = cur["low"] < range_low and cur["close"] > range_low
    reclaim_long = cur["close"] > prev["open"] and cur["close"] > prev["close"]
    bullish_trigger = _engine_bullish_trigger_marked(df_ltf_eng)

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
            strategy="alt_reclaim_v8",
            regime="mean_reversion",
            stop_loss_pct=risk / entry,
            take_profit_pct=(tp1 - entry) / entry,
            secondary_take_profit_pct=(tp2 - entry) / entry,
            tp3_pct=0.0,
            tp3_close_fraction=0.0,
            trail_pct=0.0,
            trail_atr_mult=0.0,
            tp1_close_fraction=0.6,
            tp2_close_fraction=0.4,
            be_trigger_rr=0.6,
            max_bars_override=12,
            cooldown_bars=0,
            size_multiplier=1.0,
        )

    return None


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────

def generate_signal(df, state=None, symbol=None, df_htf=None):
    if symbol == "BTC/USDT":
        return generate_signal_trend_btc(df, df_htf, symbol, state=state)
    return generate_signal_reclaim_alt(df, symbol, state=state)
