from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyState:
    trades_this_week: int = 0
    allow_shorts: bool = False
    min_adx: float = 20.0
    min_atr_rank: float = 0.30
    min_bb_rank: float = 0.30
    rsi_long: float = 55.0
    rsi_short: float = 45.0


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

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

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

    df["atr_pct_rank"] = _percent_rank(df["atr_pct"].fillna(method="ffill"), 252)
    df["bb_width_rank"] = _percent_rank(df["bb_width"].fillna(method="ffill"), 252)
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


def _long_signal(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState):
    if len(df_ltf) < 260 or len(df_htf) < 40:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, prev = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]
    htf_prev = df_htf.iloc[-2]

    trend_ok = (
        cur["close"] > cur["ema200"]
        and cur["ema50"] > cur["ema200"]
        and cur["ema20"] > cur["ema50"]
        and htf_cur["close"] > htf_cur["ema200"]
        and htf_cur["ema20"] >= htf_cur["ema50"]
        and htf_cur["ema20"] >= htf_prev["ema20"]
        and _safe_float(cur["adx"], 0.0) >= state.min_adx
    )

    vol_ok = (
        _safe_float(cur["atr_pct_rank"], 0.0) >= state.min_atr_rank
        and _safe_float(cur["bb_width_rank"], 0.0) >= state.min_bb_rank
    )

    breakout = _safe_float(cur["close"], 0.0) > _swing_high(df_ltf.iloc[:-1], 20)
    pullback = (
        _safe_float(cur["low"], 0.0) <= _safe_float(cur["ema20"], 0.0) * 1.01
        and _safe_float(cur["close"], 0.0) > _safe_float(cur["ema20"], 0.0)
        and _safe_float(prev["close"], 0.0) <= _safe_float(prev["ema20"], 0.0)
    )
    structure_ok = breakout or pullback

    momentum_ok = (
        _safe_float(cur["rsi"], 50.0) >= state.rsi_long
        and _safe_float(cur["macd_hist"], 0.0) > 0.0
        and _safe_float(cur.get("range_pos", 0.5), 0.5) >= 0.70
    )

    if not (trend_ok and vol_ok and structure_ok and momentum_ok):
        return None

    entry = _safe_float(cur["close"])
    atr = _safe_float(cur["atr"])
    swing_low = _swing_low(df_ltf.iloc[:-1], 20)
    stop_struct = min(swing_low * 0.9975, entry - (1.5 * atr))
    risk = entry - stop_struct
    if risk <= 0:
        return None

    tp1 = entry + (1.5 * risk)
    tp2 = entry + (3.0 * risk)
    tp3 = entry + (5.0 * risk)
    be_rr = 2.0
    runner = _safe_float(cur["adx"], 0.0) >= 28.0 and _safe_float(cur["rsi"], 0.0) >= 60.0

    return Signal(
        side="LONG",
        entry_price=entry,
        stop_loss=stop_struct,
        take_profit=tp1,
        symbol=symbol,
        strategy="btc_daily_mf_v1",
        regime="trend",
        confidence=0.75 if runner else 0.60,
        stop_loss_pct=risk / entry,
        take_profit_pct=(tp1 - entry) / entry,
        secondary_take_profit_pct=(tp2 - entry) / entry,
        tp3_pct=(tp3 - entry) / entry,
        tp3_close_fraction=0.25,
        trail_pct=0.0,
        trail_atr_mult=2.0,
        tp1_close_fraction=0.25,
        tp2_close_fraction=0.50,
        be_trigger_rr=be_rr,
        max_bars_override=30,
        cooldown_bars=0,
        size_multiplier=1.2 if runner else 1.0,
    )


def _short_signal(df_ltf: pd.DataFrame, df_htf: pd.DataFrame, symbol: str, state: StrategyState):
    if not state.allow_shorts or len(df_ltf) < 260 or len(df_htf) < 40:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)
    if df_ltf is None or df_htf is None:
        return None

    cur, prev = _daily_context(df_ltf)
    htf_cur = df_htf.iloc[-1]
    htf_prev = df_htf.iloc[-2]

    trend_ok = (
        cur["close"] < cur["ema200"]
        and cur["ema50"] < cur["ema200"]
        and cur["ema20"] < cur["ema50"]
        and htf_cur["close"] < htf_cur["ema200"]
        and htf_cur["ema20"] <= htf_cur["ema50"]
        and htf_cur["ema20"] <= htf_prev["ema20"]
        and _safe_float(cur["adx"], 0.0) >= state.min_adx
    )

    vol_ok = (
        _safe_float(cur["atr_pct_rank"], 0.0) >= state.min_atr_rank
        and _safe_float(cur["bb_width_rank"], 0.0) >= state.min_bb_rank
    )

    breakdown = _safe_float(cur["close"], 0.0) < _swing_low(df_ltf.iloc[:-1], 20)
    rejection = (
        _safe_float(cur["high"], 0.0) >= _safe_float(cur["ema20"], 0.0) * 0.99
        and _safe_float(cur["close"], 0.0) < _safe_float(cur["ema20"], 0.0)
        and _safe_float(prev["close"], 0.0) >= _safe_float(prev["ema20"], 0.0)
    )
    structure_ok = breakdown or rejection

    momentum_ok = (
        _safe_float(cur["rsi"], 50.0) <= state.rsi_short
        and _safe_float(cur["macd_hist"], 0.0) < 0.0
        and _safe_float(cur.get("range_pos", 0.5), 0.5) <= 0.30
    )

    if not (trend_ok and vol_ok and structure_ok and momentum_ok):
        return None

    entry = _safe_float(cur["close"])
    atr = _safe_float(cur["atr"])
    swing_high = _swing_high(df_ltf.iloc[:-1], 20)
    stop_struct = max(swing_high * 1.0025, entry + (1.5 * atr))
    risk = stop_struct - entry
    if risk <= 0:
        return None

    tp1 = entry - (1.5 * risk)
    tp2 = entry - (3.0 * risk)
    tp3 = entry - (5.0 * risk)
    be_rr = 2.0
    runner = _safe_float(cur["adx"], 0.0) >= 28.0 and _safe_float(cur["rsi"], 0.0) <= 40.0

    return Signal(
        side="SHORT",
        entry_price=entry,
        stop_loss=stop_struct,
        take_profit=tp1,
        symbol=symbol,
        strategy="btc_daily_mf_v1",
        regime="trend",
        confidence=0.75 if runner else 0.60,
        stop_loss_pct=risk / entry,
        take_profit_pct=(entry - tp1) / entry,
        secondary_take_profit_pct=(entry - tp2) / entry,
        tp3_pct=(entry - tp3) / entry,
        tp3_close_fraction=0.25,
        trail_pct=0.0,
        trail_atr_mult=2.0,
        tp1_close_fraction=0.25,
        tp2_close_fraction=0.50,
        be_trigger_rr=be_rr,
        max_bars_override=30,
        cooldown_bars=0,
        size_multiplier=1.2 if runner else 1.0,
    )


def generate_signal(df, state=None, symbol=None, df_htf=None):
    if df is None or df.empty:
        return None

    state = state or StrategyState()
    symbol = symbol or "BTC/USDT"
    if df_htf is None or df_htf.empty:
        return None

    if symbol == "BTC/USDT":
        long_sig = _long_signal(df, df_htf, symbol, state)
        if long_sig is not None:
            return long_sig
        return _short_signal(df, df_htf, symbol, state)

    return _long_signal(df, df_htf, symbol, state)
