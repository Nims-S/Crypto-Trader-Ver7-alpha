from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


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


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _has_core_indicators(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty and all(c in df.columns for c in ("atr", "rolling_body", "ema20", "ema50", "ema200"))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["rolling_body"] = (df["close"] - df["open"] ).abs().rolling(20).mean()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df if _has_core_indicators(df) else compute_indicators(df)


def _swing_low(df, n=20):
    return float(df["low"].iloc[-n:].min())


def _swing_high(df, n=20):
    return float(df["high"].iloc[-n:].max())


# BTC TREND ENGINE V3

def generate_signal_trend_btc(df_ltf, df_htf, symbol):
    if df_ltf is None or df_htf is None or len(df_ltf) < 80 or len(df_htf) < 80:
        return None

    df_ltf = _prepare(df_ltf)
    df_htf = _prepare(df_htf)

    cur = df_ltf.iloc[-1]
    prev = df_ltf.iloc[-2]
    htf = df_htf.iloc[-2]

    body = abs(float(cur["close"] - cur["open"]))
    body_ok = body >= float(cur["rolling_body"]) * 1.05 if pd.notna(cur["rolling_body"]) else False
    atr_ok = float(cur["atr"]) > float(cur["close"]) * 0.0025 if pd.notna(cur["atr"]) else False

    htf_up = htf["close"] > htf["ema200"] and htf["ema20"] > htf["ema50"]
    ltf_up = cur["ema20"] > cur["ema50"]

    pullback = cur["low"] <= cur["ema20"] * 1.01
    reclaim = cur["close"] > cur["ema20"] and prev["close"] <= prev["ema20"]
    prev_highs = df_ltf["high"].iloc[-11:-1]
    prev_lows = df_ltf["low"].iloc[-11:-1]

    breakout = cur["close"] > prev_highs.max()

    entry_condition = (pullback and reclaim) or breakout

    if htf_up and ltf_up and body_ok and atr_ok and entry_condition:
        entry = float(cur["close"])
        recent = df_ltf.iloc[:-1]
        stop = min(_swing_low(recent, 20), float(cur["ema50"])) * 0.998
        risk = entry - stop
        if risk > 0:
            return Signal(
                side="LONG",
                entry_price=entry,
                stop_loss=stop,
                take_profit=entry + risk * 2.2,
                symbol=symbol,
                strategy="btc_trend_v3",
                regime="trend",
                stop_loss_pct=risk / entry,
                take_profit_pct=(risk * 2.2) / entry,
                secondary_take_profit_pct=(risk * 4.0) / entry,
                tp1_close_fraction=0.25,
                tp2_close_fraction=0.75,
            )

    htf_down = htf["close"] < htf["ema200"] and htf["ema20"] < htf["ema50"]
    ltf_down = cur["ema20"] < cur["ema50"]

    pullback_s = cur["high"] >= cur["ema20"] * 0.99
    reclaim_s = cur["close"] < cur["ema20"] and prev["close"] >= prev["ema20"]
    prev_highs = df_ltf["high"].iloc[-11:-1]
    prev_lows = df_ltf["low"].iloc[-11:-1]
    breakdown = cur["close"] < prev_lows.min()

    entry_condition_s = (pullback_s and reclaim_s) or breakdown

    if htf_down and ltf_down and body_ok and atr_ok and entry_condition_s:
        entry = float(cur["close"])
        stop = max(_swing_high(df_ltf, 20), float(cur["ema50"])) * 1.002
        risk = stop - entry
        if risk > 0:
            return Signal(
                side="SHORT",
                entry_price=entry,
                stop_loss=stop,
                take_profit=entry - risk * 2.2,
                symbol=symbol,
                strategy="btc_trend_v3",
                regime="trend",
                stop_loss_pct=risk / entry,
                take_profit_pct=(risk * 2.2) / entry,
                secondary_take_profit_pct=(risk * 4.0) / entry,
                tp1_close_fraction=0.25,
                tp2_close_fraction=0.75,
            )

    return None


# ALT ENGINE (unchanged)

def generate_signal_reclaim_alt(df_ltf, symbol):
    if df_ltf is None or len(df_ltf) < 50:
        return None

    df_ltf = _prepare(df_ltf)
    cur = df_ltf.iloc[-1]

    range_low = df_ltf["low"].iloc[-20:].min()

    if cur["low"] < range_low and cur["close"] > range_low:
        entry = float(cur["close"])
        stop = float(cur["low"])
        risk = entry - stop
        if risk > 0:
            return Signal(
                side="LONG",
                entry_price=entry,
                stop_loss=stop,
                take_profit=entry + risk * 2.2,
                symbol=symbol,
                strategy="alt_reclaim_v1",
                regime="mean_reversion",
                stop_loss_pct=risk / entry,
                take_profit_pct=(risk * 2.2) / entry,
                secondary_take_profit_pct=(risk * 4) / entry,
                tp1_close_fraction=0.25,
                tp2_close_fraction=0.75,
            )

    return None


def generate_signal(df, state=None, symbol=None, df_htf=None):
    if symbol == "BTC/USDT":
        return generate_signal_trend_btc(df, df_htf, symbol)
    return generate_signal_reclaim_alt(df, symbol)
