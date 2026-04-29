"""Strategy scoring and promotion rules (enhanced)."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, List
from datetime import datetime
import random

from backtest import fetch_ohlcv_full, run_backtest
from strategy_registry import upsert_strategy, list_strategies


@dataclass(frozen=True)
class ScoreDecision:
    score: float
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_metrics(
    metrics: dict[str, Any],
    *,
    min_trades: int = 5,
    min_profit_factor: float = 1.10,
    min_win_rate: float = 0.45,
    max_drawdown_pct: float = -15.0,
) -> ScoreDecision:
    trades = int(metrics.get("trades", 0) or 0)
    raw_profit_factor = _safe_float(metrics.get("profit_factor", 0.0))
    profit_factor = raw_profit_factor if isfinite(raw_profit_factor) else 0.0
    profit_factor = _clamp(profit_factor, 0.0, 20.0)
    win_rate = _safe_float(metrics.get("win_rate", 0.0))
    drawdown = _safe_float(metrics.get("max_drawdown_pct", 0.0))
    return_pct = _safe_float(metrics.get("return_pct", 0.0))
    avg_rr = _safe_float(metrics.get("avg_rr_realised", 0.0))

    reasons: list[str] = []

    if trades < min_trades:
        reasons.append(f"trades<{min_trades}")
    if profit_factor < min_profit_factor:
        reasons.append(f"pf<{min_profit_factor:.2f}")
    if win_rate < min_win_rate:
        reasons.append(f"wr<{min_win_rate:.2f}")
    if drawdown <= max_drawdown_pct:
        reasons.append(f"dd<={max_drawdown_pct:.1f}")

    pf_component = min(max((profit_factor - 1.0) / 1.5, 0.0), 1.0)
    wr_component = min(max((win_rate - 0.35) / 0.45, 0.0), 1.0)
    dd_component = min(max((abs(drawdown) - 5.0) / 20.0, 0.0), 1.0)
    ret_component = min(max(return_pct / 25.0, -1.0), 1.0)
    rr_component = min(max(avg_rr / 4.0, 0.0), 1.0)
    trade_component = min(max(trades / float(max(min_trades, 1)), 0.0), 2.0) / 2.0

    density_penalty = 1.0 - min(trades / 50.0, 1.0)

    score = (
        0.28 * pf_component
        + 0.22 * wr_component
        + 0.18 * (1.0 - dd_component)
        + 0.12 * max(ret_component, 0.0)
        + 0.10 * rr_component
        + 0.10 * trade_component
        - 0.10 * density_penalty
    )

    passed = len(reasons) == 0 and score >= 0.55
    return ScoreDecision(score=score, passed=passed, reasons=tuple(reasons))


def promotion_status(decision: ScoreDecision) -> str:
    return "active" if decision.passed else "candidate"


# ------------------ WALK-FORWARD VALIDATION ------------------


def walk_forward_validate(symbol: str, timeframe: str, strategy_override: dict | None = None, folds: int = 2):
    df = fetch_ohlcv_full(symbol, timeframe)
    if df is None or df.empty or len(df) < 300:
        return {"passed": False, "score": 0.0, "reason": "insufficient_data"}

    n = len(df)
    segment = max(120, n // (folds + 2))

    scores = []

    for f in range(folds):
        train_end = segment * (f + 1)
        val_end = train_end + segment
        test_end = val_end + segment

        if test_end >= n:
            break

        idx = df.index

        train = run_backtest(symbol, timeframe, start=str(idx[0]), end=str(idx[train_end]), strategy_override=strategy_override)
        val = run_backtest(symbol, timeframe, start=str(idx[train_end]), end=str(idx[val_end]), strategy_override=strategy_override)
        test = run_backtest(symbol, timeframe, start=str(idx[val_end]), end=str(idx[test_end]), strategy_override=strategy_override)

        s_train = score_metrics(train).score
        s_val = score_metrics(val).score
        s_test = score_metrics(test).score

        scores.append((s_train + s_val + s_test) / 3.0)

    if not scores:
        return {"passed": False, "score": 0.0, "reason": "no_folds"}

    composite = mean(scores)
    spread = pstdev(scores) if len(scores) > 1 else 0.0

    robustness = max(0.0, composite - spread)

    return {
        "passed": composite > 0.55 and spread < 0.2,
        "score": composite,
        "robustness": robustness,
        "spread": spread,
    }


# ------------------ EVOLUTION LOOP ------------------


def mutate_parameters(base: dict) -> dict:
    params = dict(base or {})

    # simple intelligent mutations
    params["min_adx"] = max(10, params.get("min_adx", 20) + random.choice([-5, 0, 5]))
    params["min_atr_rank"] = max(0.2, params.get("min_atr_rank", 0.6) + random.choice([-0.1, 0, 0.1]))
    params["min_bb_rank"] = max(0.2, params.get("min_bb_rank", 0.6) + random.choice([-0.1, 0, 0.1]))
    params["rsi_long"] = min(70, max(40, params.get("rsi_long", 55) + random.choice([-5, 0, 5])))

    return params


def evolve_once(symbol: str, timeframe: str, top_k: int = 3):
    parents = list_strategies(active_only=True)[:top_k]

    results = []

    for p in parents:
        base_params = p.get("parameters") or {}

        for _ in range(2):
            child_params = mutate_parameters(base_params)

            wf = walk_forward_validate(symbol, timeframe, {"parameters": child_params})

            strategy_id = f"{symbol.replace('/', '_')}_{timeframe}_{datetime.utcnow().timestamp()}"

            upsert_strategy(
                strategy_id,
                base_strategy=p.get("strategy_id"),
                parameters=child_params,
                metrics={"wf_score": wf.get("score")},
                status="active" if wf.get("passed") else "candidate",
                active=wf.get("passed"),
                robustness_score=wf.get("robustness", 0.0),
                regime_profile="auto",
            )

            results.append({
                "parent": p.get("strategy_id"),
                "child": strategy_id,
                "wf": wf,
            })

    return results
