import random
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class StrategyCandidate:
    strategy_id: str
    base_strategy: str
    version: int
    parameters: Dict[str, Any]
    symbol: str
    timeframe: str
    tags: list
    source: str
    notes: str = ""


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def enforce_min_activity(params, feedback):
    mean_train = _safe_float((feedback or {}).get("mean_train_trades", 0), 0.0)
    mean_val = _safe_float((feedback or {}).get("mean_val_trades", 0), 0.0)
    mean_test = _safe_float((feedback or {}).get("mean_test_trades", 0), 0.0)
    score_spread = _safe_float((feedback or {}).get("score_spread", 0), 0.0)

    # Hard anti-starvation (fold-aware)
    if min(mean_train, mean_val, mean_test) < 5 or score_spread > 0.25:
        params["min_bb_rank"] = max(0.01, _safe_float(params.get("min_bb_rank", 0.1), 0.1) * 0.5)
        params["min_atr_rank"] = max(0.01, _safe_float(params.get("min_atr_rank", 0.1), 0.1) * 0.5)
        params["min_adx"] = max(5.0, _safe_float(params.get("min_adx", 10), 10) - 5.0)

        params["use_structure_filter"] = False
        params["use_reclaim_filter"] = False
        params["use_volume_filter"] = False

        # RSI loosening
        params["rsi_long"] = max(45, _safe_float(params.get("rsi_long", 55), 55) - 5)
        params["rsi_short"] = min(55, _safe_float(params.get("rsi_short", 45), 45) + 5)

    # Train ok but val/test dead → remove HTF rigidity
    if mean_train >= 5 and min(mean_val, mean_test) < 5:
        params["use_htf_filter"] = False
        if random.random() < 0.5:
            params["entry_mode"] = random.choice(["breakout", "mean_reversion"])

    return params


def mutate_parent(parent, symbol, timeframe, n_children=4, seed=None, feedback=None):
    random.seed(seed)
    children = []

    base_params = parent.get("parameters") if parent else {}

    for _ in range(n_children):
        params = dict(base_params)

        if feedback:
            params = enforce_min_activity(params, feedback)

        # Diversity injection (critical for exploration)
        if random.random() < 0.3:
            params["use_htf_filter"] = False

        if random.random() < 0.3:
            params["entry_mode"] = random.choice(["breakout", "mean_reversion"])

        # Penalize unstable configs (high variance across folds)
        if _safe_float((feedback or {}).get("score_spread", 0), 0.0) > 0.25:
            params["use_structure_filter"] = False
            if random.random() < 0.5:
                params["use_reclaim_filter"] = False

        child = StrategyCandidate(
            strategy_id=f"evo_{symbol.replace('/','_').lower()}_{timeframe}_{random.randint(1,999999)}",
            base_strategy=parent.get("strategy_id") if parent else "seed",
            version=(parent.get("version", 0) + 1) if parent else 1,
            parameters=params,
            symbol=symbol,
            timeframe=timeframe,
            tags=[symbol, timeframe, "evo"],
            source="evolution",
        )
        children.append(child)

    return children


def seed_strategy(symbol, timeframe, family="evo"):
    return StrategyCandidate(
        strategy_id=f"{family}_{symbol.replace('/','_').lower()}_{timeframe}_{random.randint(1,999999)}",
        base_strategy="seed",
        version=1,
        parameters={},
        symbol=symbol,
        timeframe=timeframe,
        tags=[symbol, timeframe, family],
        source="seed",
    )
