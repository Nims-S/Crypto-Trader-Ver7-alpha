from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd

from evolution import promotion_status, score_metrics
from strategy import StrategyState, compute_indicators, generate_signal
from strategy_registry import record_experiment, upsert_strategy

# ... (rest unchanged above) ...


def _maybe_log_experiment(args, result):
    strategy_id = args.strategy_id or f"{args.symbol.replace('/', '_').lower()}_{args.timeframe}_{'short' if args.allow_shorts else 'long'}"
    decision = score_metrics(result)

    regime_profile = "trend" if result.get("avg_rr_realised", 0) > 1 else "mean_reversion"

    registry_payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "allow_shorts": bool(args.allow_shorts),
        "max_bars": int(args.max_bars or 0),
        "decision": decision.as_dict(),
    }

    experiment = record_experiment(
        strategy_id,
        symbol=args.symbol,
        timeframe=args.timeframe,
        run_type="backtest",
        parameters=registry_payload,
        metrics={**result, "decision": decision.as_dict()},
        passed=decision.passed,
        notes="auto-logged from backtest.py",
    )

    upsert_strategy(
        strategy_id,
        base_strategy=args.base_strategy or strategy_id,
        version=int(args.version or 1),
        status=promotion_status(decision),
        parameters=registry_payload,
        metrics={**result, "decision": decision.as_dict()},
        tags=[args.symbol, args.timeframe, "backtest"],
        source="backtest",
        notes=f"decision={'pass' if decision.passed else 'fail'}",
        active=decision.passed,
        regime_profile=regime_profile,
        robustness_score=decision.score,
    )

    return {"decision": decision.as_dict(), "experiment": experiment}

# ... rest unchanged ...
