"""Runtime strategy selection for the live bot.

This module stays deliberately narrow:
- choose an active registry candidate when allowed
- reject unsafe candidates
- merge safe filter overrides into the live StrategyState
- fall back to the legacy hardcoded route when nothing qualifies
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Tuple

from strategy import StrategyState
from strategy_registry import list_strategies


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _norm_tags(tags: Any) -> set[str]:
    if not tags:
        return set()
    if isinstance(tags, str):
        return {tags.strip().lower()}
    try:
        return {str(t).strip().lower() for t in tags if str(t).strip()}
    except Exception:
        return set()


def is_strategy_safe(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False

    metrics = meta.get("metrics") or {}
    decision = metrics.get("decision") or {}

    if not bool(meta.get("active", False)):
        return False
    if not bool(decision.get("passed", False)):
        return False
    if int(metrics.get("trades", 0) or 0) < 30:
        return False
    if _safe_float(metrics.get("max_drawdown_pct", 0.0)) <= -20.0:
        return False
    if _safe_float(decision.get("score", 0.0)) < 0.55:
        return False
    return True


def select_strategy(symbol: str, timeframe: str, fallback: str) -> tuple[str, dict[str, Any] | None]:
    """Select the best safe active registry candidate for a symbol/timeframe.

    Returns a tuple of (strategy_id_used, strategy_meta). If no registry candidate
    qualifies, the fallback string is returned with None metadata.
    """
    symbol_key = (symbol or "").strip().lower()
    tf_key = (timeframe or "").strip().lower()

    strategies = list_strategies(active_only=True)
    candidates: list[dict[str, Any]] = []
    for row in strategies:
        tags = _norm_tags(row.get("tags"))
        if symbol_key in tags and tf_key in tags and is_strategy_safe(row):
            candidates.append(row)

    if not candidates:
        return fallback, None

    def _rank(row: dict[str, Any]):
        metrics = row.get("metrics") or {}
        decision = metrics.get("decision") or {}
        score = _safe_float(decision.get("score", 0.0))
        version = _safe_int(row.get("version", 0), 0)
        updated_at = row.get("updated_at") or row.get("created_at") or ""
        return (score, version, updated_at)

    best = sorted(candidates, key=_rank, reverse=True)[0]
    return best.get("strategy_id", fallback), best


def build_effective_state(base_state: StrategyState | None, strategy_meta: dict[str, Any] | None) -> StrategyState:
    """Merge only safe entry filters into a StrategyState copy."""
    state = deepcopy(base_state) if base_state is not None else StrategyState()
    if not strategy_meta:
        return state

    params = strategy_meta.get("parameters") or {}

    if "min_adx" in params:
        state.min_adx = _safe_float(params.get("min_adx", state.min_adx), state.min_adx)
    if "min_atr_rank" in params:
        state.min_atr_rank = _safe_float(params.get("min_atr_rank", state.min_atr_rank), state.min_atr_rank)
    if "min_bb_rank" in params:
        state.min_bb_rank = _safe_float(params.get("min_bb_rank", state.min_bb_rank), state.min_bb_rank)
    if "rsi_long" in params:
        state.rsi_long = _safe_float(params.get("rsi_long", state.rsi_long), state.rsi_long)
    if "rsi_short" in params:
        state.rsi_short = _safe_float(params.get("rsi_short", state.rsi_short), state.rsi_short)
    if "allow_shorts" in params:
        state.allow_shorts = bool(params.get("allow_shorts", state.allow_shorts))

    return state


def strategy_runtime_summary(strategy_meta: dict[str, Any] | None) -> dict[str, Any]:
    if not strategy_meta:
        return {"strategy_id_used": None, "strategy_score": 0.0, "was_override_used": False}

    metrics = strategy_meta.get("metrics") or {}
    decision = metrics.get("decision") or {}
    return {
        "strategy_id_used": strategy_meta.get("strategy_id"),
        "strategy_score": _safe_float(decision.get("score", 0.0)),
        "was_override_used": True,
    }
