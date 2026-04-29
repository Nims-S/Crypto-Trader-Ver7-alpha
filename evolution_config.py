"""Central configuration for the evolution pipeline.

Keeping thresholds in one place makes the search policy easier to tune and
prevents the orchestration layer from accumulating magic numbers.
"""

from __future__ import annotations

TRADE_DENSITY_BASE = {
    "1d": 5,
    "12h": 6,
    "8h": 7,
    "4h": 8,
    "2h": 10,
    "1h": 12,
    "30m": 14,
    "15m": 16,
}

# Walk-forward validation policy
DEFAULT_FOLDS = 3
DEFAULT_LOOKBACK_DAYS = 720
MIN_SPLIT_TRADES = 2
MIN_PROFIT_FACTOR = 1.00
MIN_WIN_RATE = 0.30
MAX_DRAWDOWN_PCT = -25.0
SOFT_DENSITY_FLOOR = 0.15

# Promotion policy: these are intentionally softer than the per-split scoring.
PROMOTION_MIN_SCORE = 0.50
PROMOTION_MAX_SCORE_SPREAD = 0.40
PROMOTION_MIN_VALIDATION_MEAN = 0.40
PROMOTION_MIN_TEST_MEAN = 0.40

# Exploration policy
DEFAULT_CHILDREN_PER_PARENT = 10
MAX_CHILDREN_PER_PARENT = 20

# Backtest / validation stability
MIN_WALKFORWARD_DAYS = 90
FALLBACK_TRAIN_RATIO = 0.60
FALLBACK_VAL_RATIO = 0.20
FALLBACK_TEST_RATIO = 0.20
