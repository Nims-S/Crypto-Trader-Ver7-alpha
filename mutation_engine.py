"""Strategy mutation helpers for the automated evolution loop.

The mutation space now explores structural archetypes in addition to entry
thresholds. Safe mutation targets include:
- entry model: mean reversion / trend pullback / breakout
- regime flags: use HTF filter, use volume filter, use reclaim filter
- state thresholds: ADX / ATR rank / BB rank / RSI bands
- directionality: long-only vs optional shorts

This version adds feedback-aware mutation: the engine reads parent diagnostics
and automatically biases future children toward higher trade density or tighter
filters depending on the observed failure mode.

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
    "min_adx": (4.0, 28.0),
    "min_atr_rank": (0.0, 0.35),
    "min_bb_rank": (0.0, 0.35),
    "rsi_long": (44.0, 66.0),
    "rsi_short": (34.0, 56.0),
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
        "min_adx": 8.5,
        "min_atr_rank": 0.03,
        "min_bb_rank": 0.03,
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
        "min_adx": 11.5,
        "min_atr_rank": 0.06,
        "min_bb_rank": 0.06,
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
        "min_adx": 9.5,
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
        "min_adx": 7.5,
        "min_atr_rank": 0.03,
        "min_bb_rank": 0.03,
        "rsi_long": 50.0,
        "rsi_short": 50.0,
    },
)

SPARSE_REASONS = {"sparse_signals", "zero_trades", "no_learning_signal", "trades<5", "trades<3", "floor_failed"}
WIDE_REASONS = {"too_many_signals", "overtrading", "noisy"}
RISK_REASONS = {"max_drawdown", "dd", "loss_streak", "equity_drop"}


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


def _normalize_reason_counts(feedback: dict[str, Any] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not feedback:
        return counts
    raw = feedback.get("top_fail_reasons") or feedback.get("reasons") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                counts[str(k).strip().lower()] = int(v)
            except Exception:
                counts[str(k).strip().lower()] = 0
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            key = str(item).strip().lower()
            if key:
                counts[key] = counts.get(key, 0) + 1
    elif raw:
        key = str(raw).strip().lower()
        if key:
            counts[key] = 1
    return counts


def derive_feedback_profile(parent: dict[str, Any] | None, feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Translate diagnostics / rejection reasons into an exploration profile.

    The profile is intentionally small and bounded so we can steer the mutation
    space without collapsing it into a single hard-coded strategy.
    """
    profile: dict[str, Any] = {
        "mode": "balanced",
        "force_loose": False,
        "force_breakout": False,
        "force_mean_reversion": False,
        "force_trend": False,
        "boost_shorts": False,
        "loosen_thresholds": False,
        "tighten_risk_bias": False,
        "target_min_adx": None,
        "target_min_atr_rank": None,
        "target_min_bb_rank": None,
        "target_use_htf_filter": None,
        "target_use_volume_filter": None,
        "target_use_reclaim_filter": None,
        "target_use_structure_filter": None,
        "target_use_trend_filter": None,
        "target_use_breakout_filter": None,
        "target_allow_shorts": None,
    }

    counts = _normalize_reason_counts(feedback)
    density = {}
    try:
        density = (feedback or {}).get("trade_activity") or {}
    except Exception:
        density = {}

    zero_folds = 0
    mean_density = None
    if isinstance(density, dict):
        mean_density = ((density.get("mean") or {}) if isinstance(density.get("mean"), dict) else {}).get("test")
        zero_folds = _safe_float(((density.get("zero_folds") or {}) if isinstance(density.get("zero_folds"), dict) else {}).get("test", 0), 0)

    if any(k in counts for k in SPARSE_REASONS) or (mean_density is not None and _safe_float(mean_density, 0.0) < 3.0) or zero_folds >= 1:
        profile.update(
            {
                "mode": "high_freq",
                "force_loose": True,
                "force_mean_reversion": True,
                "boost_shorts": True,
                "loosen_thresholds": True,
                "target_min_adx": 7.0,
                "target_min_atr_rank": 0.02,
                "target_min_bb_rank": 0.02,
                "target_use_htf_filter": False,
                "target_use_volume_filter": False,
                "target_use_reclaim_filter": True,
                "target_use_structure_filter": False,
                "target_use_trend_filter": False,
                "target_use_breakout_filter": False,
                "target_allow_shorts": True,
            }
        )
        if parent and str((parent.get("parameters") or {}).get("entry_mode", "")).lower() == "breakout":
            profile["force_breakout"] = True

    if any(k in counts for k in WIDE_REASONS):
        profile.update(
            {
                "mode": "trend",
                "force_trend": True,
                "target_use_htf_filter": True,
                "target_use_volume_filter": True,
                "target_use_structure_filter": True,
                "target_use_trend_filter": True,
                "target_use_breakout_filter": False,
                "target_allow_shorts": bool((parent or {}).get("allow_shorts", False)),
            }
        )

    if any(k in counts for k in RISK_REASONS):
        profile.update(
            {
                "mode": "balanced",
                "tighten_risk_bias": True,
                "target_use_htf_filter": True,
                "target_use_volume_filter": True,
                "target_use_structure_filter": True,
                "target_use_trend_filter": True,
                "target_use_breakout_filter": True,
            }
        )

    # Symbol-aware defaults: BTC favors trend, alts need more frequency.
    symbol = str((parent or {}).get("symbol") or "").upper()
    if symbol == "BTC/USDT":
        profile.setdefault("force_trend", True)
        if profile["mode"] != "high_freq":
            profile["mode"] = "trend"
            profile["target_use_htf_filter"] = True
            profile["target_use_volume_filter"] = True
    else:
        if profile["mode"] == "balanced":
            profile["mode"] = "high_freq"
            profile["force_loose"] = True

    return profile


def _apply_profile(params: dict[str, Any], profile: dict[str, Any], rng: random.Random) -> None:
    # Core structural steering.
    if profile.get("force_breakout"):
        params["entry_mode"] = "breakout"
    elif profile.get("force_mean_reversion"):
        params["entry_mode"] = "mean_reversion"
    elif profile.get("force_trend"):
        params["entry_mode"] = "trend_pullback"
    else:
        params["entry_mode"] = _mutate_choice(rng, params.get("entry_mode", "mean_reversion"), ENTRY_MODE_CHOICES)

    # Gate overrides from feedback.
    for key in (
        "use_htf_filter",
        "use_volume_filter",
        "use_reclaim_filter",
        "use_structure_filter",
        "use_trend_filter",
        "use_breakout_filter",
        "allow_shorts",
    ):
        target = profile.get(f"target_{key}")
        if target is not None:
            params[key] = bool(target)

    # High-frequency exploration pushes filters open when signals are sparse.
    if profile.get("force_loose"):
        params["use_htf_filter"] = False if rng.random() < 0.85 else params.get("use_htf_filter", False)
        params["use_volume_filter"] = False if rng.random() < 0.65 else params.get("use_volume_filter", False)
        params["use_structure_filter"] = False if rng.random() < 0.60 else params.get("use_structure_filter", False)
        params["use_trend_filter"] = False if rng.random() < 0.70 else params.get("use_trend_filter", False)
        params["use_breakout_filter"] = False if rng.random() < 0.70 else params.get("use_breakout_filter", False)
        params["allow_shorts"] = True if rng.random() < 0.55 else params.get("allow_shorts", False)

    # Structural rhythm: each profile still explores some variation.
    if rng.random() < 0.35:
        params["use_reclaim_filter"] = not bool(params.get("use_reclaim_filter", True))
    if rng.random() < 0.25:
        params["use_breakout_filter"] = bool(params.get("entry_mode") == "breakout") or bool(params.get("use_breakout_filter", False))

    # Threshold steering.
    if profile.get("loosen_thresholds"):
        params["min_adx"] = _clamp(_safe_float(params.get("min_adx", 10.0), 10.0) - rng.uniform(1.0, 3.0), *PARAM_BOUNDS["min_adx"])
        params["min_atr_rank"] = _clamp(_safe_float(params.get("min_atr_rank", 0.05), 0.05) - rng.uniform(0.00, 0.03), *PARAM_BOUNDS["min_atr_rank"])
        params["min_bb_rank"] = _clamp(_safe_float(params.get("min_bb_rank", 0.05), 0.05) - rng.uniform(0.00, 0.03), *PARAM_BOUNDS["min_bb_rank"])
    elif profile.get("tighten_risk_bias"):
        params["min_adx"] = _clamp(_safe_float(params.get("min_adx", 14.0), 14.0) + rng.uniform(0.0, 2.0), *PARAM_BOUNDS["min_adx"])
        params["min_atr_rank"] = _clamp(_safe_float(params.get("min_atr_rank", 0.10), 0.10) + rng.uniform(0.0, 0.03), *PARAM_BOUNDS["min_atr_rank"])
        params["min_bb_rank"] = _clamp(_safe_float(params.get("min_bb_rank", 0.10), 0.10) + rng.uniform(0.0, 0.03), *PARAM_BOUNDS["min_bb_rank"])
    else:
        params["min_adx"] = _clamp(_safe_float(params.get("min_adx", 12.0), 12.0) + rng.uniform(-1.5, 1.5), *PARAM_BOUNDS["min_adx"])
        params["min_atr_rank"] = _clamp(_safe_float(params.get("min_atr_rank", 0.08), 0.08) + rng.uniform(-0.02, 0.02), *PARAM_BOUNDS["min_atr_rank"])
        params["min_bb_rank"] = _clamp(_safe_float(params.get("min_bb_rank", 0.08), 0.08) + rng.uniform(-0.02, 0.02), *PARAM_BOUNDS["min_bb_rank"])

    # RSI bands remain bounded.
    params["rsi_long"] = _clamp(_safe_float(params.get("rsi_long", 52.0), 52.0) + rng.uniform(-2.0, 2.0), *PARAM_BOUNDS["rsi_long"])
    params["rsi_short"] = _clamp(_safe_float(params.get("rsi_short", 48.0), 48.0) + rng.uniform(-2.0, 2.0), *PARAM_BOUNDS["rsi_short"])

    # Extra push for high frequency mode.
    if profile.get("mode") == "high_freq":
        params["use_htf_filter"] = False if rng.random() < 0.80 else params["use_htf_filter"]
        params["use_volume_filter"] = False if rng.random() < 0.50 else params["use_volume_filter"]
        params["allow_shorts"] = True if rng.random() < 0.70 else params["allow_shorts"]
    elif profile.get("mode") == "trend":
        params["use_htf_filter"] = True
        params["use_volume_filter"] = True
        params["use_structure_filter"] = True
        params["use_trend_filter"] = True

    params["min_adx"] = _clamp(_safe_float(params.get("min_adx", 12.0), 12.0), *PARAM_BOUNDS["min_adx"])
    params["min_atr_rank"] = _clamp(_safe_float(params.get("min_atr_rank", 0.08), 0.08), *PARAM_BOUNDS["min_atr_rank"])
    params["min_bb_rank"] = _clamp(_safe_float(params.get("min_bb_rank", 0.08), 0.08), *PARAM_BOUNDS["min_bb_rank"])


def mutate_parent(
    parent: dict[str, Any] | None,
    *,
    symbol: str,
    timeframe: str,
    n_children: int = 4,
    seed: int | None = None,
    feedback: dict[str, Any] | None = None,
) -> list[MutationSpec]:
    """Create a bounded set of variants derived from a registry strategy.

    Feedback from diagnostics or failed validation is used to choose whether the
    next generation should be more permissive (to recover sparse signals) or more
    selective (to reduce noisy / risky overtrading).
    """
    rng = random.Random(seed)
    parent = parent or {}
    base_strategy = str(parent.get("base_strategy") or parent.get("strategy_id") or "seed")
    version = int(parent.get("version") or 1)
    base_params = _base_parameters(parent)
    feedback_profile = derive_feedback_profile(parent, feedback)

    children: list[MutationSpec] = []
    for idx in range(max(1, n_children)):
        params = copy.deepcopy(base_params)

        # Pick a profile that intentionally varies frequency and structure.
        if symbol == "BTC/USDT":
            profile_pool = (HIGH_FREQ_PROFILES[1], HIGH_FREQ_PROFILES[2], HIGH_FREQ_PROFILES[3])
        else:
            profile_pool = (HIGH_FREQ_PROFILES[0], HIGH_FREQ_PROFILES[1], HIGH_FREQ_PROFILES[2], HIGH_FREQ_PROFILES[3])

        profile = copy.deepcopy(profile_pool[idx % len(profile_pool)])
        # Blend feedback into the base profile to keep exploration adaptive.
        if feedback_profile.get("mode") == "high_freq":
            profile["name"] = f"{profile['name']}_feedback_loose"
            profile["entry_mode"] = feedback_profile.get("force_breakout") and "breakout" or profile.get("entry_mode", "mean_reversion")
        elif feedback_profile.get("mode") == "trend":
            profile["name"] = f"{profile['name']}_feedback_trend"
            profile["entry_mode"] = "trend_pullback"
            profile["use_htf_filter"] = True
            profile["use_volume_filter"] = True
            profile["use_structure_filter"] = True
        elif feedback_profile.get("mode") == "balanced":
            profile["name"] = f"{profile['name']}_feedback_balanced"

        _apply_profile(params, profile, rng)

        # Force direct feedback overrides last so they win.
        for key in (
            "target_min_adx",
            "target_min_atr_rank",
            "target_min_bb_rank",
            "target_use_htf_filter",
            "target_use_volume_filter",
            "target_use_reclaim_filter",
            "target_use_structure_filter",
            "target_use_trend_filter",
            "target_use_breakout_filter",
            "target_allow_shorts",
        ):
            target = feedback_profile.get(key)
            if target is not None:
                param_key = key.replace("target_", "")
                params[param_key] = target

        # Rotate archetype for extra diversity around the chosen profile.
        if feedback_profile.get("force_breakout"):
            params["entry_mode"] = "breakout"
        elif feedback_profile.get("force_mean_reversion"):
            params["entry_mode"] = "mean_reversion"
        elif feedback_profile.get("force_trend"):
            params["entry_mode"] = "trend_pullback"
        elif idx % 3 == 0:
            params["entry_mode"] = profile["entry_mode"]
        else:
            params["entry_mode"] = _mutate_choice(rng, params.get("entry_mode", "mean_reversion"), ENTRY_MODE_CHOICES)

        # Structural filters: feedback can open the gates when signals are sparse.
        if feedback_profile.get("force_loose"):
            params["use_htf_filter"] = False if rng.random() < 0.90 else params.get("use_htf_filter", False)
            params["use_volume_filter"] = False if rng.random() < 0.65 else params.get("use_volume_filter", False)
            params["use_structure_filter"] = False if rng.random() < 0.55 else params.get("use_structure_filter", False)
            params["use_trend_filter"] = False if rng.random() < 0.75 else params.get("use_trend_filter", False)
            params["use_breakout_filter"] = False if rng.random() < 0.70 else params.get("use_breakout_filter", False)
            params["allow_shorts"] = True
        else:
            if rng.random() < 0.55:
                params["use_htf_filter"] = bool(params.get("use_htf_filter", True))
            if rng.random() < 0.45:
                params["use_volume_filter"] = True
            if rng.random() < 0.40:
                params["use_reclaim_filter"] = True
            if rng.random() < 0.35:
                params["use_structure_filter"] = True
            if rng.random() < 0.30:
                params["use_trend_filter"] = bool(params.get("use_trend_filter", True))
            if rng.random() < 0.35:
                params["use_breakout_filter"] = bool(params.get("use_breakout_filter", False)) or params["entry_mode"] == "breakout"

        # Keep shorts opt-in, but let more children explore them.
        if symbol != "BTC/USDT" and rng.random() < 0.35:
            params["allow_shorts"] = True
        elif symbol == "BTC/USDT":
            params["allow_shorts"] = bool(parent.get("allow_shorts", False)) or rng.random() < 0.20

        # Extra softness for higher-frequency exploration.
        if params["entry_mode"] == "mean_reversion":
            params["min_adx"] = _clamp(params["min_adx"] - rng.uniform(0.0, 3.0), *PARAM_BOUNDS["min_adx"])
            params["use_htf_filter"] = False if rng.random() < 0.70 else params["use_htf_filter"]
            params["use_breakout_filter"] = False if rng.random() < 0.80 else params["use_breakout_filter"]
        elif params["entry_mode"] == "breakout":
            params["use_breakout_filter"] = True
            params["use_volume_filter"] = True
            params["min_atr_rank"] = _clamp(params["min_atr_rank"] + rng.uniform(0.0, 0.05), *PARAM_BOUNDS["min_atr_rank"])
        else:  # trend_pullback
            params["use_trend_filter"] = True
            params["use_structure_filter"] = True
            params["min_bb_rank"] = _clamp(params["min_bb_rank"] + rng.uniform(0.0, 0.04), *PARAM_BOUNDS["min_bb_rank"])

        # If feedback strongly indicates risk, tighten a bit instead of loosening.
        if feedback_profile.get("tighten_risk_bias"):
            params["use_htf_filter"] = True
            params["use_volume_filter"] = True
            params["use_structure_filter"] = True
            params["use_trend_filter"] = True
            params["use_breakout_filter"] = True

        payload = {
            "base_strategy": base_strategy,
            "version": version + 1,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "idx": idx,
            "profile": profile["name"],
            "feedback_mode": feedback_profile.get("mode", "balanced"),
        }
        suffix = _stable_suffix(payload)
        strategy_id = f"{base_strategy}_{symbol.replace('/', '_').lower()}_{timeframe}_{version + 1}_{suffix}"
        tags = [symbol.lower(), timeframe.lower(), "mutation", base_strategy.lower(), str(params["entry_mode"]).lower(), profile["name"], feedback_profile.get("mode", "balanced")]
        notes = (
            f"mutated_from={parent.get('strategy_id', base_strategy)} "
            f"idx={idx} profile={profile['name']} mode={params['entry_mode']} "
            f"feedback={feedback_profile.get('mode', 'balanced')}"
        )
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
