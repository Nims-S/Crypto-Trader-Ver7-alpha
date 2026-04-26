from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyState:
    trades_this_week: int = 0
    allow_shorts: bool = False
    min_adx: float = 16.0
    min_atr_rank: float = 0.15
    min_bb_rank: float = 0.15
    rsi_long: float = 53.0
    rsi_short: float = 47.0


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


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd_hist(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _percent_rank(series: pd.Series, window: int = 252) -> pd.Series:
    def _rank(x: pd.Series) -> float:
        last = x.iloc[-1]
        return float((x <= last).mean())

    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=False)


def _has_core_indicators(df: pd.DataFrame) -> bool:
    required = {
        "atr",
        "atr_pct",
        "atr_pct_rank",
        "bb_width",
        "bb_width_rank",
        "rolling_body",
        "ema20",
        "ema50",
        "ema200",
        "adx",
        "rsi",
        "macd_hist",
    }
    return df is not None and not df.empty and required.issubset(df.columns)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["atr_pct"] = df["atr"] / close.replace(0.0, np.nan)
    df["rolling_body"] = (df["close"] - df["open"]).abs().rolling(20, min_periods=20).mean()
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi"] = _rsi(close, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd_hist(close)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_mask = (up_move > down_move) & (up_move > 0)
    minus_mask = (down_move > up_move) & (down_move > 0)
    plus_dm.loc[plus_mask] = up_move.loc[plus_mask]
    minus_dm.loc[minus_mask] = down_move.loc[minus_mask]

    atr14 = tr.rolling(14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(14, min_periods=14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14, min_periods=14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df["adx"] = dx.rolling(14, min_periods=14).mean()

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)

    atr_pct_filled = df["atr_pct"].ffill()
    bb_width_filled = df["bb_width"].ffill()
    df["atr_pct_rank"] = _percent_rank(atr_pct_filled, 252)
    df["bb_width_rank"] = _percent_rank(bb_width_filled, 252)
    df["swing_high_20"] = df["high"].rolling(20, min_periods=20).max()
    df["swing_low_20"] = df["low"].rolling(20, min_periods=20).min()
    df["range_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0.0, np.nan)

    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df if _has_core_indicators(df) else compute_indicators(df)


def _swing_low(df: pd.DataFrame, n: int = 20) -> float:
    return float(df["low"].iloc[-n:].min())


def _swing_high(df: pd.DataFrame, n: int = 20) -> float:
    return float(df["high"].iloc[-n:].max())


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _daily_context(df: pd.DataFrame):
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    return cur, prev


def _trend_bias_ok(cur, htf_cur, htf_prev, state: StrategyState) -> bool:
    adx_ok = _safe_float(cur["adx"], 0.0) >= state.min_adx
    vol_ok = (
        _safe_float(cur["atr_pct_rank"], 0.0) >= state.min_atr_rank
        and _safe_float(cur["bb_width_rank"], 0.0) >= state.min_bb_rank
    )
    return adx_ok and vol_ok


def _btc_long_signal(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState):
    if len(df_ltf) < 220 or len(df_htf) < 30:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, prev = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]
    htf_prev = df_htf.iloc[-2]

    trend_ok = (
        _safe_float(cur["close"], 0.0) > _safe_float(cur["ema200"], 0.0) * 0.995
        and _safe_float(cur["ema50"], 0.0) >= _safe_float(cur["ema200"], 0.0) * 0.995
        and _safe_float(cur["ema20"], 0.0) >= _safe_float(cur["ema50"], 0.0) * 0.995
        and _safe_float(htf_cur["close"], 0.0) > _safe_float(htf_cur["ema200"], 0.0) * 0.995
        and _safe_float(htf_cur["ema20"], 0.0) >= _safe_float(htf_cur["ema50"], 0.0) * 0.995
        and _safe_float(htf_cur["ema20"], 0.0) >= _safe_float(htf_prev["ema20"], 0.0) * 0.998
        and _safe_float(cur["adx"], 0.0) >= state.min_adx
    )

    vol_ok = _trend_bias_ok(cur, htf_cur, htf_prev, state)

    breakout = _safe_float(cur["close"], 0.0) > _swing_high(df_ltf.iloc[:-1], 10) * 1.001
    pullback = (
        _safe_float(cur["low"], 0.0) <= _safe_float(cur["ema20"], 0.0) * 1.02
        and _safe_float(cur["close"], 0.0) > _safe_float(cur["ema20"], 0.0)
        and _safe_float(prev["close"], 0.0) <= _safe_float(prev["ema20"], 0.0) * 1.01
    )
    reclaim = (
        _safe_float(cur["low"], 0.0) <= _safe_float(cur["ema20"], 0.0) * 1.005
        and _safe_float(cur["close"], 0.0) > _safe_float(prev["close"], 0.0)
        and _safe_float(cur["close"], 0.0) > _safe_float(cur["ema20"], 0.0)
    )
    structure_ok = breakout or pullback or reclaim

    rolling_body = max(_safe_float(cur["rolling_body"], 0.0), 1e-9)
    momentum_ok = (
        _safe_float(cur["rsi"], 50.0) >= state.rsi_long
        and _safe_float(cur["macd_hist"], 0.0) >= -0.10 * rolling_body
        and _safe_float(cur.get("range_pos", 0.5), 0.5) >= 0.58
    )

    if not (trend_ok and vol_ok and structure_ok and momentum_ok):
        return None

    entry = _safe_float(cur["close"])
    atr = _safe_float(cur["atr"])
    swing_low = _swing_low(df_ltf.iloc[:-1], 20)
    stop_struct = min(swing_low * 0.997, entry - (1.35 * atr))
    risk = entry - stop_struct
    if risk <= 0:
        return None

    tp1 = entry + (1.35 * risk)
    tp2 = entry + (3.25 * risk)
    tp3 = entry + (6.0 * risk)
    be_rr = 2.25
    runner = _safe_float(cur["adx"], 0.0) >= 24.0 and _safe_float(cur["rsi"], 0.0) >= 58.0

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop_struct,
        take_profit=tp1,
        symbol=symbol,
        strategy="btc_daily_mf_v4",
        regime="trend",
        confidence=0.78 if runner else 0.62,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp3_pct=(tp3 - entry) / entry,
        tp3_close_fraction=0.55,
        trail_pct=0.0,
        trail_atr_mult=1.8,
        tp1_close_fraction=0.12,
        tp2_close_fraction=0.28,
        be_trigger_rr=be_rr,
        max_bars_override=50,
        cooldown_bars=0,
        size_multiplier=1.15 if runner else 1.0,
    )


def _btc_short_signal(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState):
    if not state.allow_shorts or len(df_ltf) < 220 or len(df_htf) < 30:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, prev = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]
    htf_prev = df_htf.iloc[-2]

    trend_ok = (
        _safe_float(cur["close"], 0.0) < _safe_float(cur["ema200"], 0.0) * 1.005
        and _safe_float(cur["ema50"], 0.0) <= _safe_float(cur["ema200"], 0.0) * 1.005
        and _safe_float(cur["ema20"], 0.0) <= _safe_float(cur["ema50"], 0.0) * 1.005
        and _safe_float(htf_cur["close"], 0.0) < _safe_float(htf_cur["ema200"], 0.0) * 1.005
        and _safe_float(htf_cur["ema20"], 0.0) <= _safe_float(htf_cur["ema50"], 0.0) * 1.005
        and _safe_float(htf_cur["ema20"], 0.0) <= _safe_float(htf_prev["ema20"], 0.0) * 1.002
        and _safe_float(cur["adx"], 0.0) >= state.min_adx
    )

    vol_ok = _trend_bias_ok(cur, htf_cur, htf_prev, state)

    breakdown = _safe_float(cur["close"], 0.0) < _swing_low(df_ltf.iloc[:-1], 10) * 0.999
    rejection = (
        _safe_float(cur["high"], 0.0) >= _safe_float(cur["ema20"], 0.0) * 0.995
        and _safe_float(cur["close"], 0.0) < _safe_float(cur["ema20"], 0.0)
        and _safe_float(prev["close"], 0.0) >= _safe_float(prev["ema20"], 0.0) * 0.99
    )
    structure_ok = breakdown or rejection

    rolling_body = max(_safe_float(cur["rolling_body"], 0.0), 1e-9)
    momentum_ok = (
        _safe_float(cur["rsi"], 50.0) <= state.rsi_short
        and _safe_float(cur["macd_hist"], 0.0) <= 0.10 * rolling_body
        and _safe_float(cur.get("range_pos", 0.5), 0.5) <= 0.42
    )

    if not (trend_ok and vol_ok and structure_ok and momentum_ok):
        return None

    entry = _safe_float(cur["close"])
    atr = _safe_float(cur["atr"])
    swing_high = _swing_high(df_ltf.iloc[:-1], 20)
    stop_struct = max(swing_high * 1.003, entry + (1.35 * atr))
    risk = stop_struct - entry
    if risk <= 0:
        return None

    tp1 = entry - (1.35 * risk)
    tp2 = entry - (3.25 * risk)
    tp3 = entry - (6.0 * risk)
    be_rr = 2.25
    runner = _safe_float(cur["adx"], 0.0) >= 24.0 and _safe_float(cur["rsi"], 0.0) <= 42.0

    return Signal(
        side="SHORT",
        entry_price=entry,
        stop_loss=stop_struct,
        take_profit=tp1,
        symbol=symbol,
        strategy="btc_daily_mf_v4",
        regime="trend",
        confidence=0.78 if runner else 0.62,
        stop_loss_pct=risk / entry,
        take_profit_pct=(entry - tp1) / entry,
        secondary_take_profit_pct=(entry - tp2) / entry,
        tp3_pct=(entry - tp3) / entry,
        tp3_close_fraction=0.55,
        trail_pct=0.0,
        trail_atr_mult=1.8,
        tp1_close_fraction=0.12,
        tp2_close_fraction=0.28,
        be_trigger_rr=be_rr,
        max_bars_override=50,
        cooldown_bars=0,
        size_multiplier=1.15 if runner else 1.0,
    )


def _alt_mean_reversion_long(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState):
    if len(df_ltf) < 180 or len(df_htf) < 26:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, prev = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]

    oversold = (
        _safe_float(cur["close"], 0.0) <= _safe_float(cur["bb_lower"], 0.0) * 1.03
        or _safe_float(cur["close"], 0.0) <= _safe_float(cur["ema20"], 0.0) * 0.985
        or _safe_float(cur["rsi"], 50.0) <= 44.0
    )
    reclaim = (
        _safe_float(cur["close"], 0.0) > _safe_float(cur["open"], 0.0)
        and _safe_float(cur["close"], 0.0) > _safe_float(prev["close"], 0.0) * 0.992
        and _safe_float(cur["close"], 0.0) >= _safe_float(cur["ema20"], 0.0) * 0.995
    )
    htf_filter = (
        _safe_float(htf_cur["close"], 0.0) >= _safe_float(htf_cur["ema200"], 0.0) * 0.90
        or _safe_float(htf_cur["ema20"], 0.0) >= _safe_float(htf_cur["ema50"], 0.0) * 0.96
    )
    momentum_turn = (
        _safe_float(cur["rsi"], 50.0) <= 46.0
        and _safe_float(cur["macd_hist"], 0.0) >= _safe_float(prev["macd_hist"], 0.0) * 0.90
        and _safe_float(cur.get("range_pos", 0.5), 0.5) <= 0.45
    )
    vol_ok = (
        _safe_float(cur["atr_pct_rank"], 0.0) >= max(0.08, state.min_atr_rank * 0.75)
        and _safe_float(cur["bb_width_rank"], 0.0) >= max(0.08, state.min_bb_rank * 0.75)
    )

    if not (oversold and reclaim and htf_filter and momentum_turn and vol_ok):
        return None

    entry = _safe_float(cur["close"])
    atr = _safe_float(cur["atr"])
    swing_low = _swing_low(df_ltf.iloc[:-1], 20)
    stop_struct = min(swing_low * 0.995, entry - (1.10 * atr))
    risk = entry - stop_struct
    if risk <= 0:
        return None

    tp1 = entry + (1.10 * risk)
    tp2 = entry + (2.60 * risk)
    tp3 = entry + (4.50 * risk)
    be_rr = 1.6
    strong_reversal = _safe_float(cur["rsi"], 0.0) <= 36.0 and _safe_float(cur["macd_hist"], 0.0) > 0.0

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop_struct,
        take_profit=tp1,
        symbol=symbol,
        strategy="alt_mr_v3",
        regime="mean_reversion",
        confidence=0.74 if strong_reversal else 0.62,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp3_pct=(tp3 - entry) / entry,
        tp3_close_fraction=0.25,
        trail_pct=0.0,
        trail_atr_mult=1.3,
        tp1_close_fraction=0.25,
        tp2_close_fraction=0.45,
        be_trigger_rr=be_rr,
        max_bars_override=16,
        cooldown_bars=0,
        size_multiplier=1.0 if strong_reversal else 0.9,
    )


def _alt_trend_pullback_long(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState):
    if len(df_ltf) < 180 or len(df_htf) < 26:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, prev = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]
    htf_prev = df_htf.iloc[-2]

    trend_ok = (
        _safe_float(cur["close"], 0.0) >= _safe_float(cur["ema200"], 0.0) * 0.975
        and _safe_float(cur["ema50"], 0.0) >= _safe_float(cur["ema200"], 0.0) * 0.985
        and _safe_float(htf_cur["close"], 0.0) >= _safe_float(htf_cur["ema200"], 0.0) * 0.90
        and _safe_float(htf_cur["ema20"], 0.0) >= _safe_float(htf_prev["ema20"], 0.0) * 0.995
    )

    pullback = (
        _safe_float(cur["low"], 0.0) <= _safe_float(cur["ema20"], 0.0) * 1.02
        or _safe_float(cur["close"], 0.0) <= _safe_float(cur["bb_mid"], 0.0) * 1.02
        or _safe_float(cur["close"], 0.0) <= _safe_float(cur["ema50"], 0.0) * 1.01
    )
    reclaim = (
        _safe_float(cur["close"], 0.0) >= _safe_float(cur["open"], 0.0)
        and _safe_float(cur["close"], 0.0) >= _safe_float(prev["close"], 0.0) * 0.99
        and _safe_float(cur["close"], 0.0) >= _safe_float(cur["ema20"], 0.0) * 0.99
    )
    momentum_ok = (
        46.0 <= _safe_float(cur["rsi"], 50.0) <= 68.0
        and _safe_float(cur["macd_hist"], 0.0) >= -0.15 * max(_safe_float(cur["rolling_body"], 0.0), 1e-9)
        and _safe_float(cur.get("range_pos", 0.5), 0.5) >= 0.45
    )
    vol_ok = (
        _safe_float(cur["atr_pct_rank"], 0.0) >= max(0.08, state.min_atr_rank * 0.75)
        and _safe_float(cur["bb_width_rank"], 0.0) >= max(0.08, state.min_bb_rank * 0.75)
    )

    if not (trend_ok and pullback and reclaim and momentum_ok and vol_ok):
        return None

    entry = _safe_float(cur["close"])
    atr = _safe_float(cur["atr"])
    swing_low = _swing_low(df_ltf.iloc[:-1], 20)
    stop_struct = min(swing_low * 0.995, entry - (1.20 * atr))
    risk = entry - stop_struct
    if risk <= 0:
        return None

    tp1 = entry + (1.25 * risk)
    tp2 = entry + (2.85 * risk)
    tp3 = entry + (4.75 * risk)
    strong_trend = _safe_float(cur["adx"], 0.0) >= 20.0 and _safe_float(cur["rsi"], 0.0) >= 55.0

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop_struct,
        take_profit=tp1,
        symbol=symbol,
        strategy="alt_trend_pullback_v1",
        regime="trend",
        confidence=0.68 if strong_trend else 0.58,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp3_pct=(tp3 - entry) / entry,
        tp3_close_fraction=0.20,
        trail_pct=0.0,
        trail_atr_mult=1.5,
        tp1_close_fraction=0.20,
        tp2_close_fraction=0.50,
        be_trigger_rr=2.0,
        max_bars_override=24,
        cooldown_bars=0,
        size_multiplier=0.95 if strong_trend else 0.85,
    )


def generate_signal(df, state=None, symbol=None, df_htf=None):
    if df is None or df.empty:
        return None

    state = state or StrategyState()
    symbol = symbol or "BTC/USDT"
    if df_htf is None or df_htf.empty:
        return None

    if symbol == "BTC/USDT":
        long_sig = _btc_long_signal(df, df_htf, symbol, state)
        if long_sig is not None:
            return long_sig
        return _btc_short_signal(df, df_htf, symbol, state)

    mr_sig = _alt_mean_reversion_long(df, df_htf, symbol, state)
    if mr_sig is not None:
        return mr_sig

    trend_sig = _alt_trend_pullback_long(df, df_htf, symbol, state)
    if trend_sig is not None:
        return trend_sig

    return None
