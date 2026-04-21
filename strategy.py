from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import pandas as pd


class Regime(str, Enum):
    TREND = "trend"
    NO_TRADE = "no_trade"
    HIGH_VOL = "high_vol"
    SWEEP = "sweep"


@dataclass
class StrategyState:
    trades_this_week: int = 0
    current_week_num: Optional[int] = None
    rolling_body_window: int = 20
    swing_lookback: int = 20
    atr_period: int = 14
    atr_threshold_pct: float = 0.003
    displacement_mult: float = 1.5
    sweep_depth_min: float = 0.002
    max_trades_per_week: int = 2


@dataclass
class Signal:
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    sweep_depth_pct: float
    atr: float
    symbol: str = ""
    strategy: str = "msb_sweep_v1"
    regime: str = "sweep"
    confidence: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    secondary_take_profit_pct: float = 0.0
    trail_pct: float = 0.0
    trail_atr_mult: float = 0.0
    size_multiplier: float = 1.0
    tp1_close_fraction: float = 1.0
    tp2_close_fraction: float = 0.0
    tp3_pct: float = 0.0
    tp3_enabled: bool = False
    tp3_close_fraction: float = 0.0
    cooldown_bars: int = 0
    max_trades_per_week: int = 2


TradeSignal = Signal


def _ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "timestamp" in df.columns:
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out.set_index("timestamp").sort_index()
    return df.copy()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _body_ma(df: pd.DataFrame, window: int) -> pd.Series:
    return (df["close"] - df["open"]).abs().rolling(window).mean()


def precompute(df: pd.DataFrame, state: StrategyState) -> pd.DataFrame:
    df = _ensure_index(df).copy()
    df["atr"] = _atr(df, state.atr_period)
    df["rolling_body"] = _body_ma(df, state.rolling_body_window)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return precompute(df, StrategyState())


def get_monday_range(df: pd.DataFrame, current_ts: pd.Timestamp):
    df = _ensure_index(df)
    if df.empty:
        return None, None
    current_ts = pd.Timestamp(current_ts)
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize("UTC")
    monday = df[df.index.dayofweek == 0]
    week_start = current_ts - pd.Timedelta(days=current_ts.dayofweek)
    monday = monday[monday.index.date == week_start.date()]
    if monday.empty:
        return None, None
    return float(monday["high"].max()), float(monday["low"].min())


def _args(*args: Any, **kwargs: Any):
    symbol = kwargs.pop("symbol", None)
    state = kwargs.pop("state", None)
    monday_high = kwargs.pop("monday_high", None)
    monday_low = kwargs.pop("monday_low", None)
    risk_reward = kwargs.pop("risk_reward", 2.0)
    if kwargs:
        raise TypeError(f"Unexpected kwargs: {sorted(kwargs)}")
    df = None
    if len(args) == 1:
        df = args[0]
    elif len(args) == 2:
        if isinstance(args[0], str):
            symbol, df = args
        else:
            df, state = args
    elif len(args) == 3:
        symbol, df, state = args
    elif len(args) != 0:
        raise TypeError("generate_signal expects df, (symbol, df), or (df, state)")
    return symbol, df, state, monday_high, monday_low, risk_reward


def _signal_common(symbol, side, entry, stop, tp, reason, sweep_depth_pct, atr, state):
    stop_loss_pct = abs(entry - stop) / entry
    take_profit_pct = abs(tp - entry) / entry
    return Signal(
        side=side,
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        reason=reason,
        sweep_depth_pct=round(sweep_depth_pct * 100, 3),
        atr=round(float(atr), 2),
        symbol=symbol,
        regime="sweep",
        confidence=0.92,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        secondary_take_profit_pct=take_profit_pct * 1.5,
        trail_pct=stop_loss_pct * 0.75,
        tp1_close_fraction=1.0,
        max_trades_per_week=state.max_trades_per_week,
    )


def generate_signal(*args, **kwargs):
    symbol, df, state, monday_high, monday_low, risk_reward = _args(*args, **kwargs)
    state = state if isinstance(state, StrategyState) else StrategyState()
    symbol = symbol or "BTC/USDT"
    if df is None:
        return None

    df = precompute(df, state)
    if len(df) < max(state.swing_lookback, state.rolling_body_window, state.atr_period) + 5:
        return None
    now = df.index[-1]
    week_num = int(now.isocalendar().week)
    if state.current_week_num != week_num:
        state.current_week_num = week_num
        state.trades_this_week = 0
    if state.trades_this_week >= state.max_trades_per_week:
        return None

    if monday_high is None or monday_low is None:
        monday_high, monday_low = get_monday_range(df, now)
    if monday_high is None or monday_low is None:
        return None

    cur = df.iloc[-1]
    atr = float(cur["atr"])
    avg_body = float(cur["rolling_body"])
    if pd.isna(atr) or pd.isna(avg_body):
        return None
    if atr < float(cur["close"]) * state.atr_threshold_pct:
        return None

    look = state.swing_lookback
    swing_high = float(df["high"].iloc[-(look + 1):-1].max())
    swing_low = float(df["low"].iloc[-(look + 1):-1].min())
    body_size = abs(float(cur["close"]) - float(cur["open"]))
    displaced = body_size > avg_body * state.displacement_mult

    recent_low = float(df["low"].iloc[-look:-1].min())
    sweep_long = abs(monday_low - recent_low) / monday_low
    if recent_low < monday_low and sweep_long >= state.sweep_depth_min:
        if float(cur["close"]) > monday_low and float(cur["close"]) > swing_high and displaced and float(cur["close"]) > float(cur["open"]):
            stop = recent_low * 0.998
            tp = float(cur["close"]) + (float(cur["close"]) - stop) * risk_reward
            state.trades_this_week += 1
            return _signal_common(symbol, "LONG", float(cur["close"]), stop, tp, "sweep_low + reclaim + msb + bullish_displacement", sweep_long, atr, state)

    recent_high = float(df["high"].iloc[-look:-1].max())
    sweep_short = abs(recent_high - monday_high) / monday_high
    if recent_high > monday_high and sweep_short >= state.sweep_depth_min:
        if float(cur["close"]) < monday_high and float(cur["close"]) < swing_low and displaced and float(cur["close"]) < float(cur["open"]):
            stop = recent_high * 1.002
            tp = float(cur["close"]) - (stop - float(cur["close"])) * risk_reward
            state.trades_this_week += 1
            return _signal_common(symbol, "SHORT", float(cur["close"]), stop, tp, "sweep_high + reclaim + msb + bearish_displacement", sweep_short, atr, state)

    return None
