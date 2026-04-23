from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

import pandas as pd


class Regime(str, Enum):
    TREND    = "trend"
    NO_TRADE = "no_trade"
    HIGH_VOL = "high_vol"
    SWEEP    = "sweep"


@dataclass
class StrategyState:
    # ── weekly trade limiter ─────────────────────────────────────────────────
    trades_this_week:   int            = 0
    current_week_num:   Optional[int]  = None

    # ── weekly circuit breaker ───────────────────────────────────────────────
    weekly_pnl:         float          = 0.0      # fraction of equity lost this week
    weekly_loss_limit:  float          = -0.03    # stop trading week after -3% equity loss

    # ── indicator windows ────────────────────────────────────────────────────
    rolling_body_window: int  = 20
    swing_lookback:      int  = 20
    atr_period:          int  = 14

    # ── filters ──────────────────────────────────────────────────────────────
    atr_threshold_pct:  float = 0.003   # 0.3% of price — ATR % floor
    min_atr_usd:        float = 50.0    # absolute ATR floor; gates out SOL/low-cap noise
    displacement_mult:  float = 1.5
    sweep_depth_min:    float = 0.002   # 0.2% minimum sweep depth

    # ── trade limits ─────────────────────────────────────────────────────────
    max_trades_per_week: int  = 2

    # ── Monday session window (UTC hours) ────────────────────────────────────
    monday_session_start: int = 13      # NY open
    monday_session_end:   int = 22


@dataclass
class Signal:
    side:                     str
    entry_price:              float
    stop_loss:                float
    take_profit:              float
    reason:                   str
    sweep_depth_pct:          float
    atr:                      float
    symbol:                   str   = ""
    strategy:                 str   = "msb_sweep_v2"
    regime:                   str   = "sweep"
    confidence:               float = 0.0        # intentionally 0 — not yet computed
    stop_loss_pct:            float = 0.0
    take_profit_pct:          float = 0.0
    secondary_take_profit_pct: float = 0.0
    trail_pct:                float = 0.0
    trail_atr_mult:           float = 0.0
    size_multiplier:          float = 1.0
    tp1_close_fraction:       float = 1.0
    tp2_close_fraction:       float = 0.0
    tp3_pct:                  float = 0.0
    tp3_enabled:              bool  = False
    tp3_close_fraction:       float = 0.0
    cooldown_bars:            int   = 0
    max_trades_per_week:      int   = 2
    fvg_used:                 bool  = False      # whether FVG filter was applied


TradeSignal = Signal


# ── Index helpers ─────────────────────────────────────────────────────────────

def _ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "timestamp" in df.columns:
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out.set_index("timestamp").sort_index()
    return df.copy()


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _body_ma(df: pd.DataFrame, window: int) -> pd.Series:
    return (df["close"] - df["open"]).abs().rolling(window).mean()


def precompute(df: pd.DataFrame, state: StrategyState) -> pd.DataFrame:
    df = _ensure_index(df).copy()
    df["atr"]          = _atr(df, state.atr_period)
    df["rolling_body"] = _body_ma(df, state.rolling_body_window)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper used by fetch_ohlcv_full in backtest.py."""
    return precompute(df, StrategyState())


# ── Monday range ──────────────────────────────────────────────────────────────

def get_monday_range(
    df: pd.DataFrame,
    current_ts: pd.Timestamp,
    session_start_utc: int = 13,
    session_end_utc:   int = 18,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (high, low) of Monday's NY session (13:00–18:00 UTC) for the
    current ISO week.  Falls back to full Monday if the NY session hasn't
    opened yet (e.g. signal fires on Monday morning).
    """
    df = _ensure_index(df)
    if df.empty:
        return None, None

    current_ts = pd.Timestamp(current_ts)
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize("UTC")

    week_start = current_ts - pd.Timedelta(days=current_ts.dayofweek)

    # Primary: NY session candles only
    monday_ny = df[
        (df.index.dayofweek == 0) &
        (df.index.date == week_start.date()) &
        (df.index.hour >= session_start_utc) &
        (df.index.hour <  session_end_utc)
    ]

    # Fallback: full Monday if NY session not yet available
    if monday_ny.empty:
        monday_ny = df[
            (df.index.dayofweek == 0) &
            (df.index.date == week_start.date())
        ]

    if monday_ny.empty:
        return None, None

    return float(monday_ny["high"].max()), float(monday_ny["low"].min())


# ── FVG helper ────────────────────────────────────────────────────────────────

def get_fvg(
    df: pd.DataFrame,
    direction: str,
) -> Optional[Tuple[float, float]]:
    """
    Detects the Fair Value Gap (imbalance) created by the displacement candle.

    For LONG:  gap between df[-3].high and df[-2].low  (bullish FVG)
    For SHORT: gap between df[-2].high and df[-3].low  (bearish FVG)

    Returns (fvg_high, fvg_low) if a valid, un-filled gap exists and current
    price is still trading within or below it; None otherwise.

    Caller uses fvg_low as the stop-loss anchor instead of the raw sweep low,
    giving a tighter SL and better RR.
    """
    if len(df) < 3:
        return None

    c2 = df.iloc[-3]   # pre-displacement candle
    c1 = df.iloc[-2]   # displacement candle
    c0 = df.iloc[-1]   # current signal candle

    if direction == "LONG":
        fvg_low  = float(c1["low"])
        fvg_high = float(c2["high"])
        if fvg_low >= fvg_high:
            return None                        # candles overlapped — no gap
        if float(c0["close"]) > fvg_high * 1.001:
            return None                        # price already above gap — entry missed
        return fvg_high, fvg_low

    else:  # SHORT
        fvg_high = float(c1["high"])
        fvg_low  = float(c2["low"])
        if fvg_high <= fvg_low:
            return None
        if float(c0["close"]) < fvg_low * 0.999:
            return None                        # price already below gap — entry missed
        return fvg_high, fvg_low


# ── Argument unpacking ────────────────────────────────────────────────────────

def _args(*args: Any, **kwargs: Any):
    symbol       = kwargs.pop("symbol",       None)
    state        = kwargs.pop("state",        None)
    monday_high  = kwargs.pop("monday_high",  None)
    monday_low   = kwargs.pop("monday_low",   None)
    risk_reward  = kwargs.pop("risk_reward",  2.0)
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


# ── Signal builder ────────────────────────────────────────────────────────────

def _signal_common(
    symbol, side, entry, stop, tp, reason,
    sweep_depth_pct, atr, state, fvg_used=False,
) -> Signal:
    sl_pct = abs(entry - stop) / entry
    tp_pct = abs(tp    - entry) / entry
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
        confidence=0.0,             # hardcoded 0.92 removed — not meaningful
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        secondary_take_profit_pct=tp_pct * 1.5,
        trail_pct=sl_pct * 0.75,
        tp1_close_fraction=1.0,
        max_trades_per_week=state.max_trades_per_week,
        fvg_used=fvg_used,
    )


# ── Main signal function ──────────────────────────────────────────────────────

def generate_signal(*args, **kwargs) -> Optional[Signal]:
    """
    Generates a LONG or SHORT Signal, or None.

    Conditions (all must be met):
      1. Weekly trade cap not exceeded
      2. Weekly circuit breaker not triggered (< -3% equity loss this week)
      3. ATR above % threshold AND absolute USD floor (min_atr_usd)
      4. Monday NY-session range is available
      5. Sweep of Monday high/low with minimum depth
      6. Reclaim of the swept level
      7. MSB (Market Structure Break) confirmed
      8. Directional displacement candle
      9. FVG present and price still tradeable within it (tightens SL)
    """
    symbol, df, state, monday_high, monday_low, risk_reward = _args(*args, **kwargs)
    state  = state if isinstance(state, StrategyState) else StrategyState()
    symbol = symbol or "BTC/USDT"

    if df is None:
        return None

    df = precompute(df, state)

    min_bars = max(state.swing_lookback, state.rolling_body_window, state.atr_period) + 5
    if len(df) < min_bars:
        return None

    now      = df.index[-1]
    week_num = int(now.isocalendar().week)

    # ── 1. Weekly reset ──────────────────────────────────────────────────────
    if state.current_week_num != week_num:
        state.current_week_num  = week_num
        state.trades_this_week  = 0
        state.weekly_pnl        = 0.0   # reset circuit breaker each new week

    # ── 2. Hard caps ─────────────────────────────────────────────────────────
    if state.trades_this_week >= state.max_trades_per_week:
        return None
    if state.weekly_pnl <= state.weekly_loss_limit:
        return None

    # ── 3. Monday range ──────────────────────────────────────────────────────
    if monday_high is None or monday_low is None:
        monday_high, monday_low = get_monday_range(
            df, now,
            session_start_utc=state.monday_session_start,
            session_end_utc=state.monday_session_end,
        )
    if monday_high is None or monday_low is None:
        return None

    cur      = df.iloc[-1]
    atr      = float(cur["atr"])
    avg_body = float(cur["rolling_body"])

    if pd.isna(atr) or pd.isna(avg_body):
        return None

    # ── 4. ATR filters ───────────────────────────────────────────────────────
    if atr < float(cur["close"]) * state.atr_threshold_pct:
        return None


    # ── 5. Structural pivots ─────────────────────────────────────────────────
    look       = state.swing_lookback
    swing_high = float(df["high"].iloc[-(look + 1):-1].max())
    swing_low  = float(df["low"].iloc[-(look + 1):-1].min())

    body_size  = abs(float(cur["close"]) - float(cur["open"]))
    displaced  = body_size > avg_body * state.displacement_mult

    # ── LONG setup ───────────────────────────────────────────────────────────
    recent_low  = float(df["low"].iloc[-look:-1].min())
    sweep_long  = abs(monday_low - recent_low) / monday_low

    if (recent_low < monday_low
            and sweep_long >= state.sweep_depth_min
            and float(cur["close"]) > monday_low
            and float(cur["close"]) > swing_high
            and displaced
            and float(cur["close"]) > float(cur["open"])):   # bullish candle

        fvg = get_fvg(df, "LONG")
        if fvg is not None:
            # Tighter SL anchored to FVG low
            _, fvg_low = fvg
            stop = fvg_low * 0.999
        else:
            # No FVG — fall back to sweep low anchor
            stop = recent_low * 0.998

        tp = float(cur["close"]) + (float(cur["close"]) - stop) * risk_reward
        state.trades_this_week += 1
        return _signal_common(
            symbol, "LONG", float(cur["close"]), stop, tp,
            "sweep_low+reclaim+msb+bullish_disp" + ("+fvg" if fvg is not None else ""),
            sweep_long, atr, state, fvg_used=(fvg is not None),
        )

    # ── SHORT setup ──────────────────────────────────────────────────────────
    recent_high  = float(df["high"].iloc[-look:-1].max())
    sweep_short  = abs(recent_high - monday_high) / monday_high

    if (recent_high > monday_high
            and sweep_short >= state.sweep_depth_min
            and float(cur["close"]) < monday_high
            and float(cur["close"]) < swing_low
            and displaced
            and float(cur["close"]) < float(cur["open"])):   # bearish candle

        fvg = get_fvg(df, "SHORT")
        if fvg is not None:
            fvg_high, _ = fvg
            stop = fvg_high * 1.001
        else:
            stop = recent_high * 1.002

        tp = float(cur["close"]) - (stop - float(cur["close"])) * risk_reward
        state.trades_this_week += 1
        return _signal_common(
            symbol, "SHORT", float(cur["close"]), stop, tp,
            "sweep_high+reclaim+msb+bearish_disp" + ("+fvg" if fvg is not None else ""),
            sweep_short, atr, state, fvg_used=(fvg is not None),
        )

    return None
