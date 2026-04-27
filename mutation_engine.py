"""Strategy mutation helpers for the automated evolution loop.

The mutation space now explores structural archetypes in addition to entry
thresholds. Safe mutation targets include:
- entry model: mean reversion / trend pullback / breakout
- regime flags: use HTF filter, use volume filter, use reclaim filter
- state thresholds: ADX / ATR rank / BB rank / RSI bands
- directionality: long-only vs optional shorts

Stop logic, exit logic, and position sizing remain outside the mutation space.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any


PARAM_BOUNDS = {
    "min_adx": (10.0, 28.0),
    "min_atr_rank": (0.05, 0.40),
    "min_bb_rank": (0.05, 0.40),
    "rsi_long": (48.0, 62.0),
    "rsi_short": (38.0, 52.0),
}

ENTRY_MODE_CHOICES = ("mean_reversion", "trend_pullback", "breakout")
BOOL_CHOICES = (False, True)


@dataclass(frozen=True)
class MutationSpec:
    strategy_id: str
    base_strategy: str
    version: int
    symbol: str
    timeframe: str
    parameters: dict[str, Any]
    tags: list[str]
    source: str = "mutation"
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "base_strategy": self.base_strategy,
            "version": self.version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "parameters": copy.deepcopy(self.parameters),
            "tags": list(self.tags),
            "source": self.source,
            "notes": self.notes,
        }


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _stable_suffix(payload: dict[str, Any], length: int = 8) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:length]


def _base_parameters(parent: dict[str, Any] | None) -> dict[str, Any]:
    params = {
        "allow_shorts": False,
        "entry_mode": "mean_reversion",
        "use_htf_filter": True,
        "use_volume_filter": True,
        "use_reclaim_filter": True,
        "use_structure_filter": True,
        "use_trend_filter": True,
        "use_breakout_filter": False,
        "min_adx": 16.0,
        "min_atr_rank": 0.15,
        "min_bb_rank": 0.15,
        "rsi_long": 53.0,
        "rsi_short": 47.0,
    }
    if not parent:
        return params

    raw = parent.get("parameters") or {}
    for key in params:
        if key not in raw:
            continue
        if isinstance(params[key], bool):
            params[key] = _safe_bool(raw.get(key, params[key]), params[key])
        elif isinstance(params[key], str):
            params[key] = str(raw.get(key, params[key]))
        else:
            params[key] = _safe_float(raw.get(key, params[key]), params[key])
    return params


def _mutate_choice(rng: random.Random, current: Any, choices: tuple[Any, ...]) -> Any:
    options = [choice for choice in choices if choice != current]
    return rng.choice(options) if options else current


def mutate_parent(
    parent: dict[str, Any] | None,
    *,
    symbol: str,
    timeframe: str,
    n_children: int = 4,
    seed: int | None = None,
) -> list[MutationSpec]:
    """Create a bounded set of variants derived from a registry strategy."""
    rng = random.Random(seed)
    parent = parent or {}
    base_strategy = str(parent.get("base_strategy") or parent.get("strategy_id") or "seed")
    version = int(parent.get("version") or 1)
    base_params = _base_parameters(parent)

    children: list[MutationSpec] = []
    for idx in range(max(1, n_children)):
        params = copy.deepcopy(base_params)

        # Structural exploration: rotate one of the entry archetypes per child.
        params["entry_mode"] = _mutate_choice(rng, params.get("entry_mode", "mean_reversion"), ENTRY_MODE_CHOICES)

        # Toggle structural filters in a constrained way.
        if rng.random() < 0.55:
            params["use_htf_filter"] = not bool(params.get("use_htf_filter", True)) if rng.random() < 0.5 else bool(params.get("use_htf_filter", True))
        if rng.random() < 0.45:
            params["use_volume_filter"] = not bool(params.get("use_volume_filter", True)) if rng.random() < 0.5 else bool(params.get("use_volume_filter", True))
        if rng.random() < 0.35:
            params["use_reclaim_filter"] = not bool(params.get("use_reclaim_filter", True)) if rng.random() < 0.5 else bool(params.get("use_reclaim_filter", True))
        if rng.random() < 0.35:
            params["use_structure_filter"] = not bool(params.get("use_structure_filter", True)) if rng.random() < 0.5 else bool(params.get("use_structure_filter", True))
        if rng.random() < 0.35:
            params["use_trend_filter"] = not bool(params.get("use_trend_filter", True)) if rng.random() < 0.5 else bool(params.get("use_trend_filter", True))
        if rng.random() < 0.25:
            params["use_breakout_filter"] = not bool(params.get("use_breakout_filter", False)) if rng.random() < 0.5 else bool(params.get("use_breakout_filter", False))

        # Entry thresholds remain bounded, but now complement the entry archetype.
        params["min_adx"] = _clamp(params["min_adx"] + rng.uniform(-2.5, 2.5), *PARAM_BOUNDS["min_adx"])
        params["min_atr_rank"] = _clamp(params["min_atr_rank"] + rng.uniform(-0.06, 0.06), *PARAM_BOUNDS["min_atr_rank"])
        params["min_bb_rank"] = _clamp(params["min_bb_rank"] + rng.uniform(-0.06, 0.06), *PARAM_BOUNDS["min_bb_rank"])
        params["rsi_long"] = _clamp(params["rsi_long"] + rng.uniform(-2.5, 2.5), *PARAM_BOUNDS["rsi_long"])
        params["rsi_short"] = _clamp(params["rsi_short"] + rng.uniform(-2.5, 2.5), *PARAM_BOUNDS["rsi_short"])

        # Keep shorts opt-in and only occasionally mutate them on.
        if rng.random() < 0.20:
            params["allow_shorts"] = bool(parent.get("allow_shorts", False) or rng.random() < 0.25)

        payload = {
            "base_strategy": base_strategy,
            "version": version + 1,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "idx": idx,
        }
        suffix = _stable_suffix(payload)
        strategy_id = f"{base_strategy}_{symbol.replace('/', '_').lower()}_{timeframe}_{version + 1}_{suffix}"
        tags = [symbol.lower(), timeframe.lower(), "mutation", base_strategy.lower(), str(params["entry_mode"]).lower()]
        children.append(
            MutationSpec(
                strategy_id=strategy_id,
                base_strategy=base_strategy,
                version=version + 1,
                symbol=symbol,
                timeframe=timeframe,
                parameters=params,
                tags=tags,
                notes=f"mutated_from={parent.get('strategy_id', base_strategy)} idx={idx} mode={params['entry_mode']}",
            )
        )

    return children


def seed_strategy(symbol: str, timeframe: str, *, family: str = "seed") -> MutationSpec:
    """Generate a baseline candidate when there is no parent to mutate."""
    params = _base_parameters(None)
    base_strategy = f"{family}_{symbol.replace('/', '_').lower()}"
    payload = {"base_strategy": base_strategy, "symbol": symbol, "timeframe": timeframe, "params": params}
    strategy_id = f"{base_strategy}_{timeframe}_{_stable_suffix(payload)}"
    return MutationSpec(
        strategy_id=strategy_id,
        base_strategy=base_strategy,
        version=1,
        symbol=symbol,
        timeframe=timeframe,
        parameters=params,
        tags=[symbol.lower(), timeframe.lower(), family, "seed", str(params["entry_mode"]).lower()],
        source="seed",
        notes="bootstrap candidate",
    )
