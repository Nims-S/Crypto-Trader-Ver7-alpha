import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans


class PriceActionEngine:
    def __init__(self, window=3, volume_multiplier=1.5):
        """
        window: How many candles to the left/right to check for a swing high/low.
        volume_multiplier: How much higher the volume must be compared to the average to be a valid trigger.
        """
        self.window = window
        self.volume_multiplier = volume_multiplier

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make sure the dataframe has the columns this engine expects."""
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()

        # Normalize timestamp/index if needed, but keep it lightweight.
        if "timestamp" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out = out.set_index("timestamp", drop=False)

        if "is_swing_high" not in out.columns:
            out["is_swing_high"] = False
        if "is_swing_low" not in out.columns:
            out["is_swing_low"] = False

        return out

    def get_swing_highs_lows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finds mathematical peaks and valleys in the price chart."""
        df = self._prepare(df)
        if df.empty:
            return df

        highs = df["high"].values
        swing_high_indices = argrelextrema(highs, np.greater, order=self.window)[0]
        df["is_swing_high"] = False
        df.iloc[swing_high_indices, df.columns.get_loc("is_swing_high")] = True

        lows = df["low"].values
        swing_low_indices = argrelextrema(lows, np.less, order=self.window)[0]
        df["is_swing_low"] = False
        df.iloc[swing_low_indices, df.columns.get_loc("is_swing_low")] = True

        return df

    def determine_regime(self, df: pd.DataFrame) -> str:
        """Determines if the market structure is Bullish, Bearish, or Ranging."""
        df = self._prepare(df)
        if df.empty or len(df) < max(self.window * 4, 10):
            return "UNKNOWN"

        # Populate swings on demand so callers can pass raw OHLCV.
        if "is_swing_high" not in df.columns or "is_swing_low" not in df.columns:
            df = self.get_swing_highs_lows(df)
        elif not df["is_swing_high"].any() and not df["is_swing_low"].any():
            df = self.get_swing_highs_lows(df)

        swings = df[(df["is_swing_high"] == True) | (df["is_swing_low"] == True)].dropna()

        if len(swings) < 4:
            return "UNKNOWN"

        recent_highs = swings[swings["is_swing_high"]]["high"].tail(2).values
        recent_lows = swings[swings["is_swing_low"]]["low"].tail(2).values

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return "UNKNOWN"

        if recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]:
            return "BULL_TREND"
        elif recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]:
            return "BEAR_TREND"
        else:
            return "RANGING"

    def find_support_resistance(self, df: pd.DataFrame, num_levels=4) -> list:
        """Uses K-Means clustering to find where prices bunch up (S&R zones)."""
        df = self._prepare(df)
        if df.empty:
            return []

        closes = df["close"].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_levels, random_state=0, n_init=10)
        kmeans.fit(closes)
        levels = sorted([cluster[0] for cluster in kmeans.cluster_centers_])
        return levels

    def check_bullish_trigger(self, df: pd.DataFrame) -> bool:
        """Checks the LAST TWO candles for a high-volume Bullish Engulfing pattern."""
        df = self._prepare(df)
        if len(df) < 20:
            return False

        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]

        is_engulfing = (
            (curr_candle["close"] > prev_candle["open"])
            and (curr_candle["open"] < prev_candle["close"])
            and (prev_candle["close"] < prev_candle["open"])
        )

        vol_sma_20 = df["volume"].tail(20).mean()
        high_volume = curr_candle["volume"] > (vol_sma_20 * self.volume_multiplier)

        return bool(is_engulfing and high_volume)


if __name__ == "__main__":
    print("Generating dummy OHLCV data to test the Price Action Engine...")

    dates = pd.date_range("2024-01-01", periods=50, freq="1h")
    np.random.seed(42)
    base_price = 60000
    close_prices = base_price + np.random.randn(50).cumsum() * 100

    dummy_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices - np.random.rand(50) * 50,
            "high": close_prices + np.random.rand(50) * 100,
            "low": close_prices - np.random.rand(50) * 100,
            "close": close_prices,
            "volume": np.random.randint(100, 1000, 50),
        }
    )

    dummy_data.loc[48, "open"] = 61000
    dummy_data.loc[48, "close"] = 60500
    dummy_data.loc[49, "open"] = 60400
    dummy_data.loc[49, "close"] = 61100
    dummy_data.loc[49, "volume"] = 5000

    engine = PriceActionEngine(window=3)
    df_with_swings = engine.get_swing_highs_lows(dummy_data)
    total_highs = df_with_swings["is_swing_high"].sum()
    total_lows = df_with_swings["is_swing_low"].sum()
    regime = engine.determine_regime(df_with_swings)
    levels = engine.find_support_resistance(dummy_data)
    trigger = engine.check_bullish_trigger(dummy_data)

    print("\n--- PRICE ACTION ENGINE RESULTS ---")
    print(f"Swing Highs found: {total_highs}")
    print(f"Swing Lows found:  {total_lows}")
    print(f"Current Market Regime: {regime}")
    print("Detected Support/Resistance Levels:")
    for i, level in enumerate(levels):
        print(f"  Level {i+1}: ${level:.2f}")
    print(f"High-Volume Bullish Engulfing Trigger Fired? : {trigger}")
    print("-----------------------------------")