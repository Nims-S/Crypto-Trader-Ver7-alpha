"""Strategy scoring, validation, and evolution rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import mean, pstdev
from typing import Any
from collections import Counter
import random
import time

from backtest import fetch_ohlcv_full
from strategy_registry import compute_logic_hash, list_strategies, upsert_strategy


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
    min_trades: int = 20,
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


# ------------------ DIAGNOSTICS ------------------


def _summarize_backtest(metrics: dict[str, Any], *, bars_seen: int | None, stage: str) -> dict[str, Any]:
    trades = int(metrics.get("trades", 0) or 0)
    pf = _safe_float(metrics.get("profit_factor", 0.0))
    wr = _safe_float(metrics.get("win_rate", 0.0))
    dd = _safe_float(metrics.get("max_drawdown_pct", 0.0))
    ret = _safe_float(metrics.get("return_pct", 0.0))

    density = None
    if bars_seen and bars_seen > 0:
        density = round((trades / float(bars_seen)) * 100.0, 4)

    if trades == 0:
        bottleneck = "signal_starvation"
    elif trades < 10:
        bottleneck = "sparse_signals"
    elif pf < 1.10 or wr < 0.45:
        bottleneck = "weak_edge"
    elif dd <= -15.0:
        bottleneck = "risk_instability"
    elif abs(ret) < 0.5 and trades > 0:
        bottleneck = "flat_equity"
    else:
        bottleneck = "healthy"

    return {
        "stage": stage,
        "bars_seen": bars_seen,
        "trade_density_per_100_bars": density,
        "bottleneck": bottleneck,
        "trades": trades,
        "pf": round(pf, 4),
        "win_rate": round(wr, 4),
        "drawdown": round(dd, 4),
        "return_pct": round(ret, 4),
    }


def _summarize_walk_forward(folds: list[dict[str, Any]]) -> dict[str, Any]:
    if not folds:
        return {"bottlenecks": [], "zero_trade_folds": 0, "mean_trade_density": 0.0}

    counter = Counter()
    zero_trade = 0
    densities = []

    for f in folds:
        for split in ("train", "val", "test"):
            diag = f.get(split, {}).get("diagnostics", {})
            counter[diag.get("bottleneck", "unknown")] += 1
            if diag.get("trades", 0) == 0:
                zero_trade += 1
            if diag.get("trade_density_per_100_bars") is not None:
                densities.append(diag.get("trade_density_per_100_bars"))

    return {
        "bottlenecks": counter.most_common(),
        "zero_trade_folds": zero_trade,
        "mean_trade_density": round(sum(densities) / len(densities), 4) if densities else 0.0,
    }


# ------------------ WALK-FORWARD VALIDATION ------------------


def walk_forward_validate(symbol: str, timeframe: str, strategy_override: dict | None = None, folds: int = 2):
    from backtest import run_backtest

    df = fetch_ohlcv_full(symbol, timeframe)
    if df is None or df.empty or len(df) < 300:
        return {"passed": False, "score": 0.0, "reason": "insufficient_data"}

    n = len(df)
    segment = max(120, n // (folds + 2))
    scores = []
    fold_reports = []

    for f in range(folds):
        train_end = segment * (f + 1)
        val_end = train_end + segment
        test_end = val_end + segment

        if test_end >= n:
            break

        idx = df.index

        train_m = run_backtest(symbol, timeframe, start=str(idx[0]), end=str(idx[train_end]), strategy_override=strategy_override)
        val_m = run_backtest(symbol, timeframe, start=str(idx[train_end]), end=str(idx[val_end]), strategy_override=strategy_override)
        test_m = run_backtest(symbol, timeframe, start=str(idx[val_end]), end=str(idx[test_end]), strategy_override=strategy_override)

        train_diag = _summarize_backtest(train_m, bars_seen=segment, stage="train")
        val_diag = _summarize_backtest(val_m, bars_seen=segment, stage="val")
        test_diag = _summarize_backtest(test_m, bars_seen=segment, stage="test")

        s_train = score_metrics(train_m).score
        s_val = score_metrics(val_m).score
        s_test = score_metrics(test_m).score

        scores.append((s_train + s_val + s_test) / 3.0)

        fold_reports.append({
            "train": {"metrics": train_m, "diagnostics": train_diag},
            "val": {"metrics": val_m, "diagnostics": val_diag},
            "test": {"metrics": test_m, "diagnostics": test_diag},
        })

    if not scores:
        return {"passed": False, "score": 0.0, "reason": "no_folds"}

    composite = mean(scores)
    spread = pstdev(scores) if len(scores) > 1 else 0.0
    robustness = max(0.0, composite - spread)

    wf_diag = _summarize_walk_forward(fold_reports)

    return {
        "passed": composite > 0.55 and spread < 0.2,
        "score": composite,
        "robustness": robustness,
        "spread": spread,
        "diagnostics": wf_diag,
        "folds": fold_reports,
    }


# ------------------ EVOLUTION LOOP ------------------


def _normalize_tags(tags: Any) -> set[str]:
    if not tags:
        return set()
    if isinstance(tags, str):
        return {tags.strip().lower()}
    return {str(t).strip().lower() for t in tags if str(t).strip()}


def _strategy_regime(row: dict[str, Any]) -> str:
    regime = str(row.get("regime_profile") or "").strip().lower()
    if regime in {"trend", "mean_reversion", "breakout"}:
        return regime

    params = row.get("parameters") or {}
    mode = str(params.get("entry_mode", "") or "").strip().lower()
    if mode in {"trend", "trend_following"}:
        return "trend"
    if mode in {"breakout"}:
        return "breakout"
    return "mean_reversion"


def _regime_template(regime: str) -> dict[str, Any]:
    if regime == "trend":
        return {
            "entry_mode": "trend",
            "use_trend_filter": True,
            "use_htf_filter": True,
            "use_reclaim_filter": True,
            "use_structure_filter": True,
            "use_volume_filter": True,
            "use_breakout_filter": False,
            "min_adx_shift": 5,
            "min_atr_rank_shift": 0.08,
            "min_bb_rank_shift": 0.05,
            "rsi_long_shift": 4,
            "rsi_short_shift": -2,
            "allow_shorts": False,
        }
    if regime == "breakout":
        return {
            "entry_mode": "breakout",
            "use_trend_filter": True,
            "use_htf_filter": True,
            "use_reclaim_filter": True,
            "use_structure_filter": True,
            "use_volume_filter": True,
            "use_breakout_filter": True,
            "min_adx_shift": 3,
            "min_atr_rank_shift": 0.05,
            "min_bb_rank_shift": 0.06,
            "rsi_long_shift": 2,
            "rsi_short_shift": -2,
            "allow_shorts": False,
        }
    return {
        "entry_mode": "mean_reversion",
        "use_trend_filter": True,
        "use_htf_filter": True,
        "use_reclaim_filter": True,
        "use_structure_filter": True,
        "use_volume_filter": True,
        "use_breakout_filter": False,
        "min_adx_shift": -4,
        "min_atr_rank_shift": -0.05,
        "min_bb_rank_shift": -0.05,
        "rsi_long_shift": -4,
        "rsi_short_shift": 4,
        "allow_shorts": False,
    }


def _apply_regime_template(base: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    params = dict(base or {})
    params["entry_mode"] = template.get("entry_mode", params.get("entry_mode", "mean_reversion"))
    params["use_trend_filter"] = bool(template.get("use_trend_filter", params.get("use_trend_filter", True)))
    params["use_htf_filter"] = bool(template.get("use_htf_filter", params.get("use_htf_filter", True)))
    params["use_reclaim_filter"] = bool(template.get("use_reclaim_filter", params.get("use_reclaim_filter", True)))
    params["use_structure_filter"] = bool(template.get("use_structure_filter", params.get("use_structure_filter", True)))
    params["use_volume_filter"] = bool(template.get("use_volume_filter", params.get("use_volume_filter", True)))
    params["use_breakout_filter"] = bool(template.get("use_breakout_filter", params.get("use_breakout_filter", False)))
    params["allow_shorts"] = bool(template.get("allow_shorts", params.get("allow_shorts", False)))
    params["min_adx"] = max(8.0, _safe_float(params.get("min_adx", 16.0)) + _safe_float(template.get("min_adx_shift", 0.0)))
    params["min_atr_rank"] = _clamp(_safe_float(params.get("min_atr_rank", 0.15)) + _safe_float(template.get("min_atr_rank_shift", 0.0)), 0.05, 0.95)
    params["min_bb_rank"] = _clamp(_safe_float(params.get("min_bb_rank", 0.15)) + _safe_float(template.get("min_bb_rank_shift", 0.0)), 0.05, 0.95)
    params["rsi_long"] = _clamp(_safe_float(params.get("rsi_long", 53.0)) + _safe_float(template.get("rsi_long_shift", 0.0)), 35.0, 70.0)
    params["rsi_short"] = _clamp(_safe_float(params.get("rsi_short", 47.0)) + _safe_float(template.get("rsi_short_shift", 0.0)), 30.0, 65.0)
    return params


def _explore_noise(base: dict[str, Any]) -> dict[str, Any]:
    params = dict(base or {})
    params["min_adx"] = _clamp(_safe_float(params.get("min_adx", 16.0)) + random.choice([-5, -2, 0, 2, 5]), 8.0, 40.0)
    params["min_atr_rank"] = _clamp(_safe_float(params.get("min_atr_rank", 0.15)) + random.choice([-0.12, -0.05, 0.0, 0.05, 0.12]), 0.05, 0.95)
    params["min_bb_rank"] = _clamp(_safe_float(params.get("min_bb_rank", 0.15)) + random.choice([-0.12, -0.05, 0.0, 0.05, 0.12]), 0.05, 0.95)
    params["rsi_long"] = _clamp(_safe_float(params.get("rsi_long", 53.0)) + random.choice([-8, -4, 0, 4, 8]), 35.0, 70.0)
    params["rsi_short"] = _clamp(_safe_float(params.get("rsi_short", 47.0)) + random.choice([-8, -4, 0, 4, 8]), 30.0, 65.0)
    params["use_breakout_filter"] = bool(random.choice([params.get("use_breakout_filter", False), True, False]))
    return params


def _rank(row: dict[str, Any]) -> float:
    m = row.get("metrics") or {}
    wf = m.get("walk_forward") or {}
    decision = m.get("decision") or {}
    robustness = _safe_float(row.get("robustness_score", 0.0))
    return max(
        _safe_float(wf.get("score", m.get("wf_score", 0.0))),
        _safe_float(decision.get("score", 0.0)),
    ) + 0.25 * robustness


def _candidate_pool(all_strats: list[dict[str, Any]], symbol: str, timeframe: str) -> list[dict[str, Any]]:
    symbol_key = symbol.strip().lower()
    tf_key = timeframe.strip().lower()

    tagged = []
    for s in all_strats:
        tags = _normalize_tags(s.get("tags"))
        if symbol_key in tags and tf_key in tags:
            tagged.append(s)

    pool = tagged if tagged else list(all_strats)
    pool.sort(key=_rank, reverse=True)

    diversified: list[dict[str, Any]] = []
    seen_regimes: set[str] = set()
    for row in pool:
        regime = _strategy_regime(row)
        if regime not in seen_regimes:
            diversified.append(row)
            seen_regimes.add(regime)
        if len(diversified) >= max(3, 5):
            break

    for row in pool:
        if row not in diversified:
            diversified.append(row)
        if len(diversified) >= max(5, 2 * max(1, 3)):
            break

    return diversified[: max(3, top_k if (top_k := 3) else 3)]


def evolve_once(symbol: str, timeframe: str, top_k: int = 3):
    all_strats = list_strategies(active_only=False)
    parents = _candidate_pool(all_strats, symbol, timeframe)[: max(1, top_k)]

    results = []
    seen_logic_hashes: set[str] = set()

    for p in parents:
        base_params = dict(p.get("parameters") or {})
        parent_regime = _strategy_regime(p)

        mutation_specs = [
            (parent_regime, "exploit"),
            ("trend" if parent_regime != "trend" else "mean_reversion", "cross_regime"),
            ("breakout" if parent_regime != "breakout" else "trend", "structural"),
        ]

        for target_regime, label in mutation_specs:
            if label == "exploit":
                child_params = _apply_regime_template(base_params, _regime_template(target_regime))
            elif label == "cross_regime":
                child_params = _apply_regime_template(base_params, _regime_template(target_regime))
                child_params = _explore_noise(child_params)
            else:
                child_params = _apply_regime_template(base_params, _regime_template(target_regime))
                child_params["use_breakout_filter"] = True if target_regime == "breakout" else child_params.get("use_breakout_filter", False)
                child_params["use_volume_filter"] = True
                child_params = _explore_noise(child_params)

            logic_hash = compute_logic_hash(child_params)
            if logic_hash in seen_logic_hashes:
                continue
            seen_logic_hashes.add(logic_hash)

            wf = walk_forward_validate(symbol, timeframe, {"parameters": child_params})
            strategy_id = f"{symbol.replace('/', '_')}_{timeframe}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

            upsert_strategy(
                strategy_id,
                base_strategy=p.get("strategy_id"),
                parameters=child_params,
                metrics={"wf_score": wf.get("score"), "wf": wf, "parent_score": _rank(p)},
                status="active" if wf.get("passed") else "candidate",
                active=wf.get("passed"),
                robustness_score=_safe_float(wf.get("robustness", 0.0)),
                regime_profile=target_regime,
                parent_strategy_id=p.get("strategy_id"),
                tags=list(dict.fromkeys(list(p.get("tags") or []) + [symbol, timeframe, "evo", target_regime, label])),
                source="evolution",
            )

            results.append(
                {
                    "parent": p.get("strategy_id"),
                    "parent_regime": parent_regime,
                    "child": strategy_id,
                    "child_regime": target_regime,
                    "mutation": label,
                    "logic_hash": logic_hash,
                    "wf": wf,
                }
            )

    return results
