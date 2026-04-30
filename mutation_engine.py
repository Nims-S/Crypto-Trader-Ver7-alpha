import random
from dataclasses import dataclass
from typing import Any, Dict

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


def enforce_min_activity(params, feedback):
    tr = _safe_float((feedback or {}).get("mean_train_trades", 0), 0.0)
    va = _safe_float((feedback or {}).get("mean_val_trades", 0), 0.0)
    te = _safe_float((feedback or {}).get("mean_test_trades", 0), 0.0)
    sp = _safe_float((feedback or {}).get("score_spread", 0), 0.0)
    if min(tr, va, te) < 5 or sp > 0.25:
        params["min_bb_rank"] = max(0.01, _safe_float(params.get("min_bb_rank", 0.1), 0.1) * 0.5)
        params["min_atr_rank"] = max(0.01, _safe_float(params.get("min_atr_rank", 0.1), 0.1) * 0.5)
        params["min_adx"] = max(5.0, _safe_float(params.get("min_adx", 10), 10) - 5.0)
        params["use_structure_filter"] = False
        params["use_reclaim_filter"] = False
        params["use_volume_filter"] = False
        params["rsi_long"] = max(45, _safe_float(params.get("rsi_long", 55), 55) - 5)
        params["rsi_short"] = min(55, _safe_float(params.get("rsi_short", 45), 45) + 5)
    if tr >= 5 and min(va, te) < 5:
        params["use_htf_filter"] = False
        if random.random() < 0.5:
            params["entry_mode"] = random.choice(["breakout", "mean_reversion"])
    return params


def _apply_profit_repair(params, feedback, symbol):
    pf = _safe_float((feedback or {}).get("mean_test_pf", 0), 0.0)
    wr = _safe_float((feedback or {}).get("mean_test_wr", 0), 0.0)
    tr = _safe_float((feedback or {}).get("mean_test_trades", 0), 0.0)
    sp = _safe_float((feedback or {}).get("score_spread", 0), 0.0)
    if tr >= 15 and pf < 1.1:
        if wr >= 0.45:
            params["entry_mode"] = random.choice(["breakout", "trend_pullback"])
            params["use_breakout_filter"] = True
            params["use_trend_filter"] = True
            params["use_structure_filter"] = True
            params["use_volume_filter"] = True
            params["tp1_close_fraction"] = 0.20
            params["tp2_close_fraction"] = 0.30
            params["tp3_close_fraction"] = 0.25
            params["be_trigger_rr"] = 2.0
            params["trail_atr_mult"] = 1.15 if symbol.startswith("BTC") else 1.30
            params["trail_ema20"] = bool(symbol.startswith("BTC"))
            params["max_bars_override"] = 24 if symbol.startswith("BTC") else 18
            if symbol.startswith("BTC"):
                params["use_htf_filter"] = True
        else:
            params["entry_mode"] = random.choice(["mean_reversion", "breakout"])
            params["use_structure_filter"] = False
            params["use_volume_filter"] = False
            params["tp1_close_fraction"] = 0.30
            params["tp2_close_fraction"] = 0.35
            params["tp3_close_fraction"] = 0.15
            params["be_trigger_rr"] = 1.2
            params["trail_atr_mult"] = 1.45
            params["trail_ema20"] = False
            params["max_bars_override"] = 16
    if tr >= 15 and sp > 0.25:
        params["use_structure_filter"] = True
        params["use_volume_filter"] = True
        params["be_trigger_rr"] = max(_safe_float(params.get("be_trigger_rr", 1.6), 1.6), 1.8)
        params["trail_atr_mult"] = _clamp(_safe_float(params.get("trail_atr_mult", 1.4), 1.4), 1.1, 2.0)
    return params


def mutate_parent(parent, symbol, timeframe, n_children=4, seed=None, feedback=None):
    random.seed(seed)
    base_params = dict((parent or {}).get("parameters") or {})
    children = []
    for _ in range(max(1, n_children)):
        params = dict(base_params)
        if feedback:
            params = enforce_min_activity(params, feedback)
            params = _apply_profit_repair(params, feedback, symbol)
        if random.random() < 0.3:
            params["use_htf_filter"] = False
        if random.random() < 0.3:
            params["entry_mode"] = random.choice(["breakout", "mean_reversion"])
        if _safe_float((feedback or {}).get("score_spread", 0), 0.0) > 0.25:
            params["use_structure_filter"] = False
            if random.random() < 0.5:
                params["use_reclaim_filter"] = False
        if random.random() < 0.35:
            params["tp1_close_fraction"] = _clamp(_safe_float(params.get("tp1_close_fraction", 0.25), 0.25) + random.uniform(-0.05, 0.05), 0.10, 0.40)
        if random.random() < 0.35:
            params["tp2_close_fraction"] = _clamp(_safe_float(params.get("tp2_close_fraction", 0.35), 0.35) + random.uniform(-0.08, 0.08), 0.20, 0.55)
        if random.random() < 0.35:
            params["trail_atr_mult"] = _clamp(_safe_float(params.get("trail_atr_mult", 1.4), 1.4) + random.uniform(-0.15, 0.15), 1.0, 2.0)
        if random.random() < 0.25:
            params["be_trigger_rr"] = _clamp(_safe_float(params.get("be_trigger_rr", 1.6), 1.6) + random.uniform(-0.3, 0.4), 0.8, 3.0)
        sid = f"evo_{symbol.replace('/','_').lower()}_{timeframe}_{random.randint(1,999999)}"
        children.append(StrategyCandidate(sid, str((parent or {}).get("strategy_id") or "seed"), int((parent or {}).get("version", 0) or 0) + 1, params, symbol, timeframe, [symbol, timeframe, "evo"], "evolution"))
    return children


def seed_strategy(symbol, timeframe, family="evo"):
    return StrategyCandidate(f"{family}_{symbol.replace('/','_').lower()}_{timeframe}_{random.randint(1,999999)}", "seed", 1, {}, symbol, timeframe, [symbol, timeframe, family], "seed")
