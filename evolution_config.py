"""Central configuration for the evolution pipeline.

Keeping thresholds in one place makes the search policy easier to tune and
prevents the orchestration layer from accumulating magic numbers.
"""

from __future__ import annotations

TRADE_DENSITY_BASE = {
    "1d": 4,
    "12h": 5,
    "8h": 6,
    "4h": 6,
    "2h": 8,
    "1h": 10,
    "30m": 12,
    "15m": 14,
}

# Walk-forward validation policy
DEFAULT_FOLDS = 3
DEFAULT_LOOKBACK_DAYS = 720
MIN_SPLIT_TRADES = 3
MIN_PROFIT_FACTOR = 1.05
MIN_WIN_RATE = 0.40
MAX_DRAWDOWN_PCT = -20.0
SOFT_DENSITY_FLOOR = 0.20

# Promotion policy: these are intentionally tighter than the per-split scoring.
PROMOTION_MIN_SCORE = 0.55
PROMOTION_MAX_SCORE_SPREAD = 0.35
PROMOTION_MIN_VALIDATION_MEAN = 0.45
PROMOTION_MIN_TEST_MEAN = 0.45

# Exploration policy
DEFAULT_CHILDREN_PER_PARENT = 8
MAX_CHILDREN_PER_PARENT = 16

# Backtest / validation stability
MIN_WALKFORWARD_DAYS = 90
FALLBACK_TRAIN_RATIO = 0.60
FALLBACK_VAL_RATIO = 0.20
FALLBACK_TEST_RATIO = 0.20
