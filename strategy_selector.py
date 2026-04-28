"""Runtime strategy selection for the live bot (enhanced)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from strategy import StrategyState
from strategy_registry import rank_strategies


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def select_strategy(symbol: str, timeframe: str, fallback: str) -> tuple[str, dict[str, Any] | None]:
    symbol_key = (symbol or "").strip()
    tf_key = (timeframe or "").strip()

    candidates = rank_strategies(symbol=symbol_key, timeframe=tf_key, active_only=True, limit=5)

    if not candidates:
        return fallback, None

    best = candidates[0]
    return best.get("strategy_id", fallback), best


def build_effective_state(base_state: StrategyState | None, strategy_meta: dict[str, Any] | None) -> StrategyState:
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
