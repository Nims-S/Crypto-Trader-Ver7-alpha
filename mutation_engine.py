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
    "min_atr_rank": (0.03, 0.30),
    "min_bb_rank": (0.03, 0.30),
    "rsi_long": (46.0, 64.0),
    "rsi_short": (36.0, 54.0),
}

ENTRY_MODE_CHOICES = ("mean_reversion", "trend_pullback", "breakout")
HIGH_FREQ_PROFILES = (
    {
        "name": "mr_loose",
        "entry_mode": "mean_reversion",
        "allow_shorts": True,
        "use_htf_filter": False,
        "use_volume_filter": False,
        "use_reclaim_filter": True,
        "use_structure_filter": False,
        "use_trend_filter": False,
        "use_breakout_filter": False,
        "min_adx": 10.5,
        "min_atr_rank": 0.05,
        "min_bb_rank": 0.05,
        "rsi_long": 51.0,
        "rsi_short": 49.0,
    },
    {
        "name": "trend_balanced",
        "entry_mode": "trend_pullback",
        "allow_shorts": False,
        "use_htf_filter": True,
        "use_volume_filter": True,
        "use_reclaim_filter": True,
        "use_structure_filter": True,
        "use_trend_filter": True,
        "use_breakout_filter": False,
        "min_adx": 12.5,
        "min_atr_rank": 0.08,
        "min_bb_rank": 0.08,
        "rsi_long": 52.0,
        "rsi_short": 48.0,
    },
    {
        "name": "breakout_aggressive",
        "entry_mode": "breakout",
        "allow_shorts": True,
        "use_htf_filter": False,
        "use_volume_filter": True,
        "use_reclaim_filter": False,
        "use_structure_filter": False,
        "use_trend_filter": False,
        "use_breakout_filter": True,
        "min_adx": 11.0,
        "min_atr_rank": 0.04,
        "min_bb_rank": 0.04,
        "rsi_long": 54.0,
        "rsi_short": 46.0,
    },
    {
        "name": "hybrid_high_freq",
        "entry_mode": "mean_reversion",
        "allow_shorts": True,
        "use_htf_filter": False,
        "use_volume_filter": True,
        "use_reclaim_filter": True,
        "use_structure_filter": True,
        "use_trend_filter": False,
        "use_breakout_filter": True,
        "min_adx": 9.5,
        "min_atr_rank": 0.05,
        "min_bb_rank": 0.05,
        "rsi_long": 50.0,
        "rsi_short": 50.0,
    },
)


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
        "min_atr_rank": 0.12,
        "min_bb_rank": 0.12,
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


def _apply_profile(params: dict[str, Any], profile: dict[str, Any], rng: random.Random) -> None:
    for key, value in profile.items():
        if key == "name":
            continue
        if isinstance(value, bool):
            params[key] = bool(value if rng.random() > 0.20 else not value)
        elif isinstance(value, (int, float)):
            # Small jitter widens the candidate set while keeping it bounded.
            jitter = rng.uniform(-0.12, 0.12)
            params[key] = value + (value * jitter if value else jitter)
        else:
            params[key] = value

    # Strongly bias frequency upward: loosen entry gates a bit more.
    params["min_adx"] = _clamp(_safe_float(params.get("min_adx", 12.0), 12.0), *PARAM_BOUNDS["min_adx"])
    params["min_atr_rank"] = _clamp(_safe_float(params.get("min_atr_rank", 0.08), 0.08), *PARAM_BOUNDS["min_atr_rank"])
    params["min_bb_rank"] = _clamp(_safe_float(params.get("min_bb_rank", 0.08), 0.08), *PARAM_BOUNDS["min_bb_rank"])
    params["rsi_long"] = _clamp(_safe_float(params.get("rsi_long", 52.0), 52.0), *PARAM_BOUNDS["rsi_long"])
    params["rsi_short"] = _clamp(_safe_float(params.get("rsi_short", 48.0), 48.0), *PARAM_BOUNDS["rsi_short"])


def mutate_parent(
    parent: dict[str, Any] | None,
    *,
    symbol: str,
    timeframe: str,
    n_children: int = 4,
    seed: int | None = None,
) -> list[MutationSpec]:
    """Create a bounded set of variants derived from a registry strategy.

    The child set intentionally covers different trade frequencies by mixing
    looser, more active archetypes with a smaller number of conservative ones.
    """
    rng = random.Random(seed)
    parent = parent or {}
    base_strategy = str(parent.get("base_strategy") or parent.get("strategy_id") or "seed")
    version = int(parent.get("version") or 1)
    base_params = _base_parameters(parent)

    children: list[MutationSpec] = []
    for idx in range(max(1, n_children)):
        params = copy.deepcopy(base_params)

        # Pick a profile that intentionally varies frequency and structure.
        if symbol == "BTC/USDT":
            profile_pool = (HIGH_FREQ_PROFILES[1], HIGH_FREQ_PROFILES[2], HIGH_FREQ_PROFILES[3])
        else:
            profile_pool = (HIGH_FREQ_PROFILES[0], HIGH_FREQ_PROFILES[1], HIGH_FREQ_PROFILES[2], HIGH_FREQ_PROFILES[3])
        profile = profile_pool[idx % len(profile_pool)]
        _apply_profile(params, profile, rng)

        # Rotate archetype for extra diversity around the chosen profile.
        params["entry_mode"] = _mutate_choice(rng, params.get("entry_mode", "mean_reversion"), ENTRY_MODE_CHOICES)
        if idx % 3 == 0:
            params["entry_mode"] = profile["entry_mode"]

        # Structural filters: high frequency children should not all gate on HTF.
        if rng.random() < 0.70:
            params["use_htf_filter"] = bool(params.get("use_htf_filter", True)) if idx % 4 != 0 else False
        if rng.random() < 0.65:
            params["use_volume_filter"] = True
        if rng.random() < 0.45:
            params["use_reclaim_filter"] = True
        if rng.random() < 0.40:
            params["use_structure_filter"] = True
        if rng.random() < 0.35:
            params["use_trend_filter"] = bool(params.get("use_trend_filter", True)) if idx % 2 == 0 else False
        if rng.random() < 0.35:
            params["use_breakout_filter"] = bool(params.get("use_breakout_filter", False)) or params["entry_mode"] == "breakout"

        # Keep shorts opt-in, but let more children explore them.
        if symbol != "BTC/USDT" and rng.random() < 0.35:
            params["allow_shorts"] = True
        elif symbol == "BTC/USDT":
            params["allow_shorts"] = bool(parent.get("allow_shorts", False)) or rng.random() < 0.20

        # Extra softness for higher-frequency exploration.
        if params["entry_mode"] == "mean_reversion":
            params["min_adx"] = _clamp(params["min_adx"] - rng.uniform(0.0, 2.0), *PARAM_BOUNDS["min_adx"])
            params["use_htf_filter"] = False if rng.random() < 0.60 else params["use_htf_filter"]
            params["use_breakout_filter"] = False if rng.random() < 0.70 else params["use_breakout_filter"]
        elif params["entry_mode"] == "breakout":
            params["use_breakout_filter"] = True
            params["use_volume_filter"] = True
            params["min_atr_rank"] = _clamp(params["min_atr_rank"] + rng.uniform(0.0, 0.05), *PARAM_BOUNDS["min_atr_rank"])
        else:  # trend_pullback
            params["use_trend_filter"] = True
            params["use_structure_filter"] = True
            params["min_bb_rank"] = _clamp(params["min_bb_rank"] + rng.uniform(0.0, 0.04), *PARAM_BOUNDS["min_bb_rank"])

        payload = {
            "base_strategy": base_strategy,
            "version": version + 1,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "idx": idx,
            "profile": profile["name"],
        }
        suffix = _stable_suffix(payload)
        strategy_id = f"{base_strategy}_{symbol.replace('/', '_').lower()}_{timeframe}_{version + 1}_{suffix}"
        tags = [symbol.lower(), timeframe.lower(), "mutation", base_strategy.lower(), str(params["entry_mode"]).lower(), profile["name"]]
        notes = f"mutated_from={parent.get('strategy_id', base_strategy)} idx={idx} profile={profile['name']} mode={params['entry_mode']}"
        children.append(
            MutationSpec(
                strategy_id=strategy_id,
                base_strategy=base_strategy,
                version=version + 1,
                symbol=symbol,
                timeframe=timeframe,
                parameters=params,
                tags=tags,
                notes=notes,
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
