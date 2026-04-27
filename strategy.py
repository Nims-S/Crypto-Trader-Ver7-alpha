from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    trail_ema20: bool = False
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
        "bbwp",
        "rolling_body",
        "ema20",
        "ema50",
        "ema50_slope",
        "ema200",
        "sma200",
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

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["atr_pct"] = df["atr"] / close.replace(0.0, np.nan)
    df["rolling_body"] = (df["close"] - df["open"]).abs().rolling(20, min_periods=20).mean()
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["sma200"] = close.rolling(200, min_periods=200).mean()
    df["ema50_slope"] = df["ema50"].diff(3) / 3.0
    df["ema20_slope"] = df["ema20"].diff(3) / 3.0
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
    df["bbwp"] = df["bb_width_rank"]
    df["swing_high_20"] = df["high"].rolling(20, min_periods=20).max()
    df["swing_low_20"] = df["low"].rolling(20, min_periods=20).min()
    df["range_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0.0, np.nan)
    df["volume_sma20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["bbwp_low_streak"] = (df["bbwp"] < 0.10).astype(int).groupby((df["bbwp"] >= 0.10).astype(int).cumsum()).cumsum()

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


def _safe_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _daily_context(df: pd.DataFrame):
    return df.iloc[-1], df.iloc[-2]


def _trend_slope_ok(cur, bullish: bool) -> bool:
    ema50_slope = _safe_float(cur.get("ema50_slope", 0.0), 0.0)
    return ema50_slope > 0 if bullish else ema50_slope < 0


def _vol_expansion_recent(df: pd.DataFrame) -> bool:
    if len(df) < 20:
        return False
    cur = _safe_float(df.iloc[-1].get("bbwp", 0.0), 0.0)
    prev = _safe_float(df.iloc[-2].get("bbwp", 0.0), 0.0)
    recent_floor = float(df["bbwp"].iloc[-16:-1].min()) if len(df) >= 16 else float(df["bbwp"].min())
    return cur >= 0.25 and prev < 0.25 and recent_floor <= 0.10


def _in_cash_mode(df: pd.DataFrame) -> bool:
    if len(df) < 16:
        return False
    return int(df["bbwp"].iloc[-16:].lt(0.10).sum()) >= 16


def _volume_breakout_ok(df: pd.DataFrame) -> bool:
    cur = df.iloc[-1]
    vol = _safe_float(cur.get("volume", 0.0), 0.0)
    vma = _safe_float(cur.get("volume_sma20", 0.0), 0.0)
    return vma > 0 and vol >= 1.25 * vma


def _params(strategy_override: dict[str, Any] | None) -> dict[str, Any]:
    return (strategy_override or {}).get("parameters") or {}


def _entry_mode(params: dict[str, Any]) -> str:
    return str(params.get("entry_mode", "mean_reversion") or "mean_reversion").strip().lower()


def _flag(params: dict[str, Any], key: str, default: bool = True) -> bool:
    return _safe_bool(params.get(key, default), default)


def _build_state(state: StrategyState, params: dict[str, Any]) -> StrategyState:
    return StrategyState(
        trades_this_week=state.trades_this_week,
        allow_shorts=_flag(params, "allow_shorts", state.allow_shorts),
        min_adx=_safe_float(params.get("min_adx", state.min_adx), state.min_adx),
        min_atr_rank=_safe_float(params.get("min_atr_rank", state.min_atr_rank), state.min_atr_rank),
        min_bb_rank=_safe_float(params.get("min_bb_rank", state.min_bb_rank), state.min_bb_rank),
        rsi_long=_safe_float(params.get("rsi_long", state.rsi_long), state.rsi_long),
        rsi_short=_safe_float(params.get("rsi_short", state.rsi_short), state.rsi_short),
    )


def _vetf_long(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState):
    if len(df_ltf) < 220 or len(df_htf) < 30:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, _ = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]

    if _in_cash_mode(df_ltf) and not (_vol_expansion_recent(df_ltf) and _volume_breakout_ok(df_ltf)):
        return None

    trend_ok = (
        _safe_float(cur["close"], 0.0) > _safe_float(cur["sma200"], 0.0)
        and _trend_slope_ok(cur, bullish=True)
        and _safe_float(htf_cur["close"], 0.0) > _safe_float(htf_cur["sma200"], 0.0)
        and _trend_slope_ok(htf_cur, bullish=True)
    )
    if not trend_ok:
        return None

    if not (_vol_expansion_recent(df_ltf) and _volume_breakout_ok(df_ltf)):
        return None

    swing_high = _swing_high(df_ltf.iloc[:-1], 20)
    if _safe_float(cur["close"], 0.0) <= swing_high:
        return None

    entry = _safe_float(cur["close"], 0.0)
    atr = _safe_float(cur["atr"], 0.0)
    if entry <= 0 or atr <= 0:
        return None

    struct_stop = _swing_low(df_ltf.iloc[:-1], 20) * 0.999
    atr_stop = entry - (1.5 * atr)
    stop = max(struct_stop, atr_stop)
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (2.0 * risk)
    tp2 = entry + (999.0 * risk)

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp1,
        symbol=symbol,
        strategy="vetf_btc_v1",
        regime="trend",
        confidence=0.78,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp3_pct=0.0,
        tp3_close_fraction=0.0,
        trail_ema20=True,
        tp1_close_fraction=0.50,
        tp2_close_fraction=0.50,
        be_trigger_rr=0.0,
        max_bars_override=60,
        cooldown_bars=0,
        size_multiplier=1.0,
    )


def _vetf_short(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState):
    if not state.allow_shorts or len(df_ltf) < 220 or len(df_htf) < 30:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, _ = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]

    if _in_cash_mode(df_ltf) and not (_vol_expansion_recent(df_ltf) and _volume_breakout_ok(df_ltf)):
        return None

    trend_ok = (
        _safe_float(cur["close"], 0.0) < _safe_float(cur["sma200"], 0.0)
        and _trend_slope_ok(cur, bullish=False)
        and _safe_float(htf_cur["close"], 0.0) < _safe_float(htf_cur["sma200"], 0.0)
        and _trend_slope_ok(htf_cur, bullish=False)
    )
    if not trend_ok:
        return None

    if not (_vol_expansion_recent(df_ltf) and _volume_breakout_ok(df_ltf)):
        return None

    swing_low = _swing_low(df_ltf.iloc[:-1], 20)
    if _safe_float(cur["close"], 0.0) >= swing_low:
        return None

    entry = _safe_float(cur["close"], 0.0)
    atr = _safe_float(cur["atr"], 0.0)
    if entry <= 0 or atr <= 0:
        return None

    struct_stop = _swing_high(df_ltf.iloc[:-1], 20) * 1.001
    atr_stop = entry + (1.5 * atr)
    stop = min(struct_stop, atr_stop)
    if stop <= entry:
        return None

    risk = stop - entry
    tp1 = entry - (2.0 * risk)
    tp2 = entry - (999.0 * risk)

    return Signal(
        side="SHORT",
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp1,
        symbol=symbol,
        strategy="vetf_btc_v1",
        regime="trend",
        confidence=0.78,
        stop_loss_pct=risk / entry,
        take_profit_pct=(entry - tp1) / entry,
        secondary_take_profit_pct=(entry - tp2) / entry,
        tp3_pct=0.0,
        tp3_close_fraction=0.0,
        trail_ema20=True,
        tp1_close_fraction=0.50,
        tp2_close_fraction=0.50,
        be_trigger_rr=0.0,
        max_bars_override=60,
        cooldown_bars=0,
        size_multiplier=1.0,
    )


def _alt_mean_reversion_long(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState, params: dict[str, Any]):
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
        _safe_float(htf_cur["close"], 0.0) >= _safe_float(htf_cur["sma200"], 0.0) * 0.90
        or _safe_float(htf_cur["ema20"], 0.0) >= _safe_float(htf_cur["ema50"], 0.0) * 0.96
    )
    trend_bias = (
        _safe_float(cur["close"], 0.0) >= _safe_float(cur["ema200"], 0.0) * 0.95
        or _safe_float(cur["ema20"], 0.0) >= _safe_float(cur["ema50"], 0.0) * 0.96
    )
    structure_ok = _safe_float(cur.get("range_pos", 0.5), 0.5) <= 0.45 or _safe_float(cur["close"], 0.0) >= _safe_float(cur["swing_low_20"], 0.0) * 1.0
    momentum_turn = (
        _safe_float(cur["rsi"], 50.0) <= 46.0
        and _safe_float(cur["macd_hist"], 0.0) >= _safe_float(prev["macd_hist"], 0.0) * 0.90
        and _safe_float(cur.get("range_pos", 0.5), 0.5) <= 0.45
    )
    vol_ok = (
        _safe_float(cur["atr_pct_rank"], 0.0) >= max(0.08, state.min_atr_rank * 0.75)
        and _safe_float(cur["bb_width_rank"], 0.0) >= max(0.08, state.min_bb_rank * 0.75)
    )

    if _flag(params, "use_htf_filter", True) and not htf_filter:
        return None
    if _flag(params, "use_trend_filter", True) and not trend_bias:
        return None
    if _flag(params, "use_structure_filter", True) and not structure_ok:
        return None
    if _flag(params, "use_reclaim_filter", True) and not reclaim:
        return None
    if _flag(params, "use_volume_filter", True) and not vol_ok:
        return None
    if not oversold or not momentum_turn:
        return None

    entry = _safe_float(cur["close"], 0.0)
    atr = _safe_float(cur["atr"], 0.0)
    if entry <= 0 or atr <= 0:
        return None

    stop = min(_swing_low(df_ltf.iloc[:-1], 20) * 0.995, entry - (1.10 * atr))
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (1.10 * risk)
    tp2 = entry + (2.60 * risk)
    tp3 = entry + (4.50 * risk)
    strong_reversal = _safe_float(cur["rsi"], 0.0) <= 36.0 and _safe_float(cur["macd_hist"], 0.0) > 0.0

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop,
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
        trail_ema20=False,
        trail_pct=0.0,
        trail_atr_mult=1.3,
        tp1_close_fraction=0.25,
        tp2_close_fraction=0.45,
        be_trigger_rr=1.6,
        max_bars_override=16,
        cooldown_bars=0,
        size_multiplier=1.0 if strong_reversal else 0.9,
    )


def _alt_trend_pullback_long(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState, params: dict[str, Any]):
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

    if _flag(params, "use_htf_filter", True) and not trend_ok:
        return None
    if _flag(params, "use_trend_filter", True) and not trend_ok:
        return None
    if _flag(params, "use_reclaim_filter", True) and not reclaim:
        return None
    if _flag(params, "use_volume_filter", True) and not vol_ok:
        return None
    if _flag(params, "use_structure_filter", True) and not pullback:
        return None
    if not momentum_ok:
        return None

    entry = _safe_float(cur["close"], 0.0)
    atr = _safe_float(cur["atr"], 0.0)
    if entry <= 0 or atr <= 0:
        return None

    stop = min(_swing_low(df_ltf.iloc[:-1], 20) * 0.995, entry - (1.20 * atr))
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (1.25 * risk)
    tp2 = entry + (2.85 * risk)
    tp3 = entry + (4.75 * risk)
    strong_trend = _safe_float(cur["adx"], 0.0) >= 20.0 and _safe_float(cur["rsi"], 0.0) >= 55.0

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop,
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
        trail_ema20=False,
        trail_pct=0.0,
        trail_atr_mult=1.5,
        tp1_close_fraction=0.20,
        tp2_close_fraction=0.50,
        be_trigger_rr=2.0,
        max_bars_override=24,
        cooldown_bars=0,
        size_multiplier=0.95 if strong_trend else 0.85,
    )


def _alt_breakout_long(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState, params: dict[str, Any]):
    if len(df_ltf) < 180 or len(df_htf) < 26:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, prev = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]

    breakout_trigger = _safe_float(cur["close"], 0.0) >= max(_safe_float(cur["bb_upper"], 0.0), _safe_float(cur["swing_high_20"], 0.0)) * 0.998
    htf_trend_ok = _safe_float(htf_cur["close"], 0.0) >= _safe_float(htf_cur["sma200"], 0.0) * 0.90 or _safe_float(htf_cur["ema20"], 0.0) >= _safe_float(htf_cur["ema50"], 0.0) * 0.98
    trend_ok = _safe_float(cur["close"], 0.0) >= _safe_float(cur["ema200"], 0.0) * 0.97 and _safe_float(cur["ema50"], 0.0) >= _safe_float(cur["ema200"], 0.0) * 0.985
    reclaim_ok = _safe_float(cur["close"], 0.0) >= _safe_float(cur["open"], 0.0) and _safe_float(cur["close"], 0.0) >= _safe_float(prev["close"], 0.0) * 0.995
    volume_ok = _volume_breakout_ok(df_ltf)
    structure_ok = _safe_float(cur["close"], 0.0) >= _safe_float(cur["swing_high_20"], 0.0) * 0.998
    vol_regime_ok = _safe_float(cur["atr_pct_rank"], 0.0) >= max(0.08, state.min_atr_rank * 0.75) and _safe_float(cur["bb_width_rank"], 0.0) >= max(0.08, state.min_bb_rank * 0.75)

    if _flag(params, "use_breakout_filter", True) and not breakout_trigger:
        return None
    if _flag(params, "use_htf_filter", True) and not htf_trend_ok:
        return None
    if _flag(params, "use_trend_filter", True) and not trend_ok:
        return None
    if _flag(params, "use_reclaim_filter", True) and not reclaim_ok:
        return None
    if _flag(params, "use_volume_filter", True) and not volume_ok:
        return None
    if _flag(params, "use_structure_filter", True) and not structure_ok:
        return None
    if not vol_regime_ok:
        return None

    entry = _safe_float(cur["close"], 0.0)
    atr = _safe_float(cur["atr"], 0.0)
    if entry <= 0 or atr <= 0:
        return None

    stop = min(_swing_low(df_ltf.iloc[:-1], 20) * 0.995, entry - (1.35 * atr))
    if stop >= entry:
        return None

    risk = entry - stop
    tp1 = entry + (1.40 * risk)
    tp2 = entry + (3.10 * risk)
    tp3 = entry + (5.00 * risk)
    strong_breakout = breakout_trigger and volume_ok and trend_ok

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp1,
        symbol=symbol,
        strategy="alt_breakout_v1",
        regime="breakout",
        confidence=0.72 if strong_breakout else 0.60,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp3_pct=(tp3 - entry) / entry,
        tp3_close_fraction=0.20,
        trail_ema20=False,
        trail_pct=0.0,
        trail_atr_mult=1.7,
        tp1_close_fraction=0.20,
        tp2_close_fraction=0.45,
        be_trigger_rr=1.8,
        max_bars_override=18,
        cooldown_bars=0,
        size_multiplier=1.0 if strong_breakout else 0.9,
    )


def _alt_breakout_short(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState, params: dict[str, Any]):
    if not state.allow_shorts or len(df_ltf) < 180 or len(df_htf) < 26:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, prev = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]

    breakdown_trigger = _safe_float(cur["close"], 0.0) <= min(_safe_float(cur["bb_lower"], 0.0), _safe_float(cur["swing_low_20"], 0.0)) * 1.002
    htf_trend_ok = _safe_float(htf_cur["close"], 0.0) <= _safe_float(htf_cur["sma200"], 0.0) * 1.10 or _safe_float(htf_cur["ema20"], 0.0) <= _safe_float(htf_cur["ema50"], 0.0) * 1.02
    trend_ok = _safe_float(cur["close"], 0.0) <= _safe_float(cur["ema200"], 0.0) * 1.03 and _safe_float(cur["ema50"], 0.0) <= _safe_float(cur["ema200"], 0.0) * 1.015
    reclaim_ok = _safe_float(cur["close"], 0.0) <= _safe_float(cur["open"], 0.0) and _safe_float(cur["close"], 0.0) <= _safe_float(prev["close"], 0.0) * 1.005
    volume_ok = _volume_breakout_ok(df_ltf)
    structure_ok = _safe_float(cur["close"], 0.0) <= _safe_float(cur["swing_low_20"], 0.0) * 1.002
    vol_regime_ok = _safe_float(cur["atr_pct_rank"], 0.0) >= max(0.08, state.min_atr_rank * 0.75) and _safe_float(cur["bb_width_rank"], 0.0) >= max(0.08, state.min_bb_rank * 0.75)

    if _flag(params, "use_breakout_filter", True) and not breakdown_trigger:
        return None
    if _flag(params, "use_htf_filter", True) and not htf_trend_ok:
        return None
    if _flag(params, "use_trend_filter", True) and not trend_ok:
        return None
    if _flag(params, "use_reclaim_filter", True) and not reclaim_ok:
        return None
    if _flag(params, "use_volume_filter", True) and not volume_ok:
        return None
    if _flag(params, "use_structure_filter", True) and not structure_ok:
        return None
    if not vol_regime_ok:
        return None

    entry = _safe_float(cur["close"], 0.0)
    atr = _safe_float(cur["atr"], 0.0)
    if entry <= 0 or atr <= 0:
        return None

    stop = max(_swing_high(df_ltf.iloc[:-1], 20) * 1.005, entry + (1.35 * atr))
    if stop <= entry:
        return None

    risk = stop - entry
    tp1 = entry - (1.40 * risk)
    tp2 = entry - (3.10 * risk)
    tp3 = entry - (5.00 * risk)
    strong_breakdown = breakdown_trigger and volume_ok and trend_ok

    return Signal(
        side="SHORT",
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp1,
        symbol=symbol,
        strategy="alt_breakout_v1",
        regime="breakout",
        confidence=0.72 if strong_breakdown else 0.60,
        stop_loss_pct=risk / entry,
        take_profit_pct=(entry - tp1) / entry,
        secondary_take_profit_pct=(entry - tp2) / entry,
        tp3_pct=(entry - tp3) / entry,
        tp3_close_fraction=0.20,
        trail_ema20=False,
        trail_pct=0.0,
        trail_atr_mult=1.7,
        tp1_close_fraction=0.20,
        tp2_close_fraction=0.45,
        be_trigger_rr=1.8,
        max_bars_override=18,
        cooldown_bars=0,
        size_multiplier=1.0 if strong_breakdown else 0.9,
    )


def generate_signal(df, state=None, symbol=None, df_htf=None, strategy_override=None):
    if df is None or df.empty:
        return None

    state = state or StrategyState()
    symbol = symbol or "BTC/USDT"
    if df_htf is None or df_htf.empty:
        return None

    params = _params(strategy_override)
    state = _build_state(state, params)
    entry_mode = _entry_mode(params)

    if symbol == "BTC/USDT":
        long_sig = _vetf_long(df, df_htf, symbol, state)
        if long_sig is not None:
            return long_sig
        return _vetf_short(df, df_htf, symbol, state)

    if entry_mode == "breakout":
        sig = _alt_breakout_long(df, df_htf, symbol, state, params)
        if sig is not None:
            return sig
        return _alt_breakout_short(df, df_htf, symbol, state, params)

    if entry_mode == "trend_pullback":
        sig = _alt_trend_pullback_long(df, df_htf, symbol, state, params)
        if sig is not None:
            return sig
        return _alt_breakout_long(df, df_htf, symbol, state, params) if _flag(params, "use_breakout_filter", False) else None

    mr_sig = _alt_mean_reversion_long(df, df_htf, symbol, state, params)
    if mr_sig is not None:
        return mr_sig

    trend_sig = _alt_trend_pullback_long(df, df_htf, symbol, state, params)
    if trend_sig is not None:
        return trend_sig

    breakout_sig = _alt_breakout_long(df, df_htf, symbol, state, params)
    if breakout_sig is not None:
        return breakout_sig

    return _alt_breakout_short(df, df_htf, symbol, state, params)
