from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict

from feedback_engine import build_feedback_summary


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


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _apply_directives(params: dict, directives: dict, symbol: str) -> dict:
    if not directives:
        return params

    for key in ("use_htf_filter","use_volume_filter","use_structure_filter","use_reclaim_filter","use_trend_filter"):
        if key in directives:
            params[key] = directives[key]

    if "entry_mode" in directives:
        params["entry_mode"] = directives["entry_mode"]

    if "min_adx_delta" in directives:
        params["min_adx"] = max(5.0, _safe_float(params.get("min_adx", 10), 10) + directives["min_adx_delta"])
    if "min_atr_rank_multiplier" in directives:
        params["min_atr_rank"] = max(0.01, _safe_float(params.get("min_atr_rank", 0.1), 0.1) * directives["min_atr_rank_multiplier"])
    if "min_bb_rank_multiplier" in directives:
        params["min_bb_rank"] = max(0.01, _safe_float(params.get("min_bb_rank", 0.1), 0.1) * directives["min_bb_rank_multiplier"])

    for key in ("tp1_close_fraction","tp2_close_fraction","tp3_close_fraction","be_trigger_rr","trail_atr_mult","trail_ema20","max_bars_override"):
        if key in directives:
            params[key] = directives[key]

    if symbol.startswith("BTC") and directives.get("prefer_trend_pullback"):
        params["use_htf_filter"] = True

    return params


def mutate_parent(parent, symbol, timeframe, n_children=4, seed=None, feedback=None):
    rng = random.Random(seed)
    base_params = dict((parent or {}).get("parameters") or {})

    if feedback is None:
        feedback = build_feedback_summary(symbol=symbol, timeframe=timeframe)

    directives = (feedback or {}).get("mutation_directives") or {}

    children = []
    for _ in range(max(1, n_children)):
        params = dict(base_params)

        params = _apply_directives(params, directives, symbol)

        if rng.random() < 0.4:
            params["entry_mode"] = rng.choice(["breakout","mean_reversion","trend_pullback"])

        if directives.get("loosen_filters") and rng.random() < 0.5:
            params["use_structure_filter"] = False
            params["use_volume_filter"] = False

        if rng.random() < 0.35:
            params["tp1_close_fraction"] = _clamp(_safe_float(params.get("tp1_close_fraction", 0.25), 0.25) + rng.uniform(-0.05, 0.05), 0.10, 0.40)
        if rng.random() < 0.35:
            params["tp2_close_fraction"] = _clamp(_safe_float(params.get("tp2_close_fraction", 0.35), 0.35) + rng.uniform(-0.08, 0.08), 0.20, 0.55)
        if rng.random() < 0.35:
            params["trail_atr_mult"] = _clamp(_safe_float(params.get("trail_atr_mult", 1.4), 1.4) + rng.uniform(-0.15, 0.15), 1.0, 2.0)
        if rng.random() < 0.25:
            params["be_trigger_rr"] = _clamp(_safe_float(params.get("be_trigger_rr", 1.6), 1.6) + rng.uniform(-0.3, 0.4), 0.8, 3.0)

        sid = f"evo_{symbol.replace('/','_').lower()}_{timeframe}_{rng.randint(1,999999)}"
        children.append(StrategyCandidate(sid,str((parent or {}).get("strategy_id") or "seed"),int((parent or {}).get("version", 0) or 0) + 1,params,symbol,timeframe,[symbol, timeframe, "evo"],"evolution"))

    return children


def seed_strategy(symbol, timeframe, family="evo"):
    return StrategyCandidate(f"{family}_{symbol.replace('/','_').lower()}_{timeframe}_{random.randint(1,999999)}","seed",1,{},symbol,timeframe,[symbol, timeframe, family],"seed")
