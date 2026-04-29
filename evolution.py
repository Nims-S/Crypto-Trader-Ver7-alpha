"""
Evolution engine with:
- BTC 4h entry loosening
- Signal-floor diagnostics
- Trade density tracking BEFORE scoring changes
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any
import random
import time

from backtest import fetch_ohlcv_full
from strategy_registry import compute_logic_hash, list_strategies, upsert_strategy


# ------------------ SCORING ------------------

@dataclass(frozen=True)
class ScoreDecision:
    score: float
    passed: bool
    reasons: tuple[str, ...]


def _safe(v, d=0.0):
    try:
        x = float(v)
        return x
    except:
        return d


def score_metrics(m: dict) -> ScoreDecision:
    trades = int(m.get("trades", 0))
    pf = _safe(m.get("profit_factor", 0))
    wr = _safe(m.get("win_rate", 0))
    dd = _safe(m.get("max_drawdown_pct", 0))

    reasons = []
    if trades < 20:
        reasons.append("trades<20")
    if pf < 1.1:
        reasons.append("pf<1.1")
    if wr < 0.45:
        reasons.append("wr<0.45")

    score = (
        0.4 * min(pf / 2.0, 1)
        + 0.3 * wr
        + 0.2 * max(0, 1 + dd / 20)
        + 0.1 * min(trades / 40, 1)
    )

    return ScoreDecision(score, len(reasons) == 0 and score > 0.55, tuple(reasons))


# ------------------ SIGNAL FLOOR ------------------

def signal_floor_target(symbol, tf):
    if symbol.startswith("BTC") and tf == "4h":
        return 2.5  # key fix
    return 1.5


def summarize_bt(m, bars, symbol, tf):
    trades = int(m.get("trades", 0))

    density = (trades / bars) * 100 if bars else 0
    target = signal_floor_target(symbol, tf)

    if trades == 0:
        bottleneck = "signal_starvation"
    elif trades < 10:
        bottleneck = "sparse_signals"
    elif m.get("profit_factor", 0) < 1.1:
        bottleneck = "weak_edge"
    else:
        bottleneck = "ok"

    return {
        "trades": trades,
        "density": round(density, 3),
        "target": target,
        "floor_met": density >= target,
        "bottleneck": bottleneck,
    }


# ------------------ WALK FORWARD ------------------

def walk_forward_validate(symbol, tf, strategy_override=None):
    from backtest import run_backtest

    df = fetch_ohlcv_full(symbol, tf)
    if df is None or len(df) < 300:
        return {"passed": False, "score": 0}

    n = len(df)
    seg = n // 3

    results = []
    densities = []

    for i in range(2):
        train = run_backtest(symbol, tf, start=str(df.index[0]), end=str(df.index[seg]), strategy_override=strategy_override)
        val = run_backtest(symbol, tf, start=str(df.index[seg]), end=str(df.index[2*seg]), strategy_override=strategy_override)
        test = run_backtest(symbol, tf, start=str(df.index[2*seg]), end=str(df.index[-1]), strategy_override=strategy_override)

        d = summarize_bt(test, seg, symbol, tf)
        densities.append(d["density"])

        results.append({
            "train": train,
            "val": val,
            "test": test,
            "diagnostics": d
        })

    score = mean([score_metrics(r["test"]).score for r in results])
    spread = pstdev([score_metrics(r["test"]).score for r in results]) if len(results) > 1 else 0

    mean_density = mean(densities)
    target = signal_floor_target(symbol, tf)

    return {
        "passed": score > 0.55,
        "score": score,
        "spread": spread,
        "diagnostics": {
            "mean_density": round(mean_density, 3),
            "target_density": target,
            "density_gap": round(target - mean_density, 3),
        },
        "folds": results
    }


# ------------------ EVOLUTION ------------------

def btc_loosen(params):
    p = dict(params)

    # 🔥 THIS IS THE REAL FIX
    p["allow_shorts"] = True
    p["min_adx"] = max(6, p.get("min_adx", 12) - 4)
    p["min_atr_rank"] = max(0.03, p.get("min_atr_rank", 0.1) - 0.05)
    p["min_bb_rank"] = max(0.03, p.get("min_bb_rank", 0.1) - 0.05)

    # remove filters = more signals
    p["use_reclaim_filter"] = False
    p["use_structure_filter"] = False
    p["use_volume_filter"] = False

    return p


def evolve_once(symbol, tf):
    parents = list_strategies()

    results = []

    for p in parents[:3]:
        base = dict(p.get("parameters", {}))

        child = dict(base)

        # noise
        child["min_adx"] = base.get("min_adx", 12) + random.choice([-3, 0, 3])

        if symbol.startswith("BTC") and tf == "4h":
            child = btc_loosen(child)

        logic_hash = compute_logic_hash(child)

        wf = walk_forward_validate(symbol, tf, {"parameters": child})

        sid = f"{symbol.replace('/','_')}_{tf}_{int(time.time()*1000)}"

        upsert_strategy(
            sid,
            base_strategy=p.get("strategy_id"),
            parameters=child,
            metrics={"wf": wf},
            status="candidate",
            active=False,
            robustness_score=wf.get("score", 0),
            regime_profile="auto",
            parent_strategy_id=p.get("strategy_id"),
            tags=[symbol, tf, "evo"],
            source="evolution"
        )

        results.append({
            "parent": p.get("strategy_id"),
            "child": sid,
            "wf": wf
        })

    return results