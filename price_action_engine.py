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

    def get_swing_highs_lows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finds mathematical peaks and valleys in the price chart."""
        df = df.copy()
        
        # Find local peaks (Swing Highs) using the High column
        highs = df['high'].values
        swing_high_indices = argrelextrema(highs, np.greater, order=self.window)[0]
        df['is_swing_high'] = False
        df.iloc[swing_high_indices, df.columns.get_loc('is_swing_high')] = True

        # Find local valleys (Swing Lows) using the Low column
        lows = df['low'].values
        swing_low_indices = argrelextrema(lows, np.less, order=self.window)[0]
        df['is_swing_low'] = False
        df.iloc[swing_low_indices, df.columns.get_loc('is_swing_low')] = True
        
        return df

    def determine_regime(self, df: pd.DataFrame) -> str:
        """Determines if the market structure is Bullish, Bearish, or Ranging."""
        # Get only the rows where a swing high or low occurred
        swings = df[(df['is_swing_high'] == True) | (df['is_swing_low'] == True)].dropna()
        
        if len(swings) < 4:
            return "UNKNOWN" # Not enough data

        # Extract the last two swing highs and last two swing lows
        recent_highs = swings[swings['is_swing_high']]['high'].tail(2).values
        recent_lows = swings[swings['is_swing_low']]['low'].tail(2).values

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return "UNKNOWN"

        # Higher Highs AND Higher Lows
        if recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]:
            return "BULL_TREND"
        # Lower Highs AND Lower Lows
        elif recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]:
            return "BEAR_TREND"
        else:
            return "RANGING"

    def find_support_resistance(self, df: pd.DataFrame, num_levels=4) -> list:
        """Uses K-Means clustering to find where prices bunch up (S&R zones)."""
        # We cluster the Closing prices
        closes = df['close'].values.reshape(-1, 1)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=num_levels, random_state=0, n_init=10)
        kmeans.fit(closes)
        
        # Get the cluster centers (the S&R lines) and sort them
        levels = sorted([cluster[0] for cluster in kmeans.cluster_centers_])
        return levels

    def check_bullish_trigger(self, df: pd.DataFrame) -> bool:
        """Checks the LAST TWO candles for a high-volume Bullish Engulfing pattern."""
        if len(df) < 20: 
            return False # Need enough data for Volume Moving Average
            
        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]
        
        # 1. Check Bullish Engulfing Math
        is_engulfing = (
            (curr_candle['close'] > prev_candle['open']) and 
            (curr_candle['open'] < prev_candle['close']) and 
            (prev_candle['close'] < prev_candle['open']) # Prev candle was red
        )
        
        # 2. Check Volume Math (Is current volume > 1.5x the 20-period average?)
        vol_sma_20 = df['volume'].tail(20).mean()
        high_volume = curr_candle['volume'] > (vol_sma_20 * self.volume_multiplier)
        
        return is_engulfing and high_volume

# ==========================================
# TEST SCRIPT - RUN THIS TO SEE IT IN ACTION
# ==========================================
if __name__ == "__main__":
    print("Generating dummy OHLCV data to test the Price Action Engine...")
    
    # Create 50 rows of dummy market data simulating a chart
    dates = pd.date_range('2024-01-01', periods=50, freq='1h')
    np.random.seed(42)
    base_price = 60000
    close_prices = base_price + np.random.randn(50).cumsum() * 100
    
    dummy_data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - np.random.rand(50) * 50,
        'high': close_prices + np.random.rand(50) * 100,
        'low': close_prices - np.random.rand(50) * 100,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, 50)
    })
    
    # Force the last two candles to be a high-volume Bullish Engulfing pattern for testing
    dummy_data.loc[48, 'open'] = 61000; dummy_data.loc[48, 'close'] = 60500  # Red candle
    dummy_data.loc[49, 'open'] = 60400; dummy_data.loc[49, 'close'] = 61100  # Green engulfing
    dummy_data.loc[49, 'volume'] = 5000  # Massive volume spike

    # Initialize the engine
    engine = PriceActionEngine(window=3)
    
    # 1. Get Swings
    df_with_swings = engine.get_swing_highs_lows(dummy_data)
    total_highs = df_with_swings['is_swing_high'].sum()
    total_lows = df_with_swings['is_swing_low'].sum()
    
    # 2. Get Regime
    regime = engine.determine_regime(df_with_swings)
    
    # 3. Get Support/Resistance
    levels = engine.find_support_resistance(dummy_data)
    
    # 4. Check for Trigger
    trigger = engine.check_bullish_trigger(dummy_data)
    
    # Print Results
    print("\n--- PRICE ACTION ENGINE RESULTS ---")
    print(f"Swing Highs found: {total_highs}")
    print(f"Swing Lows found:  {total_lows}")
    print(f"Current Market Regime: {regime}")
    print("Detected Support/Resistance Levels:")
    for i, level in enumerate(levels):
        print(f"  Level {i+1}: ${level:.2f}")
    print(f"High-Volume Bullish Engulfing Trigger Fired? : {trigger}")
    print("-----------------------------------")