"""Fully automated strategy evolution loop.

This version is feedback-aware:
- runs walk-forward validation
- records diagnostics
- uses rejection signals and activity data to steer the next generation
- keeps registry writes and audit logging isolated behind safe helpers
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

from backtest import run_backtest
from db import init_db
from diagnostics import build_candidate_diagnostics
from mutation_engine import mutate_parent, seed_strategy
from strategy_registry import list_strategies, record_experiment, upsert_strategy, record_evolution_run
from validation import build_walk_forward_folds, default_evolution_window, summarize_walk_forward_reports

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LINK/USDT", "AVAX/USDT", "SOL/USDT"]
DEFAULT_TIMEFRAMES = ["1d", "4h"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def _parent_feedback_from_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Convert one candidate's metrics into a minimal feedback payload."""
    metrics = metrics or {}
    wf = metrics.get("walk_forward") or {}
    diag = metrics.get("diagnostics") or {}

    activity = diag.get("trade_activity") or {}
    mean_test = 0.0
    zero_folds = 0
    try:
        mean_test = _safe_float(((activity.get("mean") or {}) if isinstance(activity.get("mean"), dict) else {}).get("test", 0.0), 0.0)
        zero_folds = int((((activity.get("zero_folds") or {}) if isinstance(activity.get("zero_folds"), dict) else {}).get("test", 0)) or 0)
    except Exception:
        pass

    reasons = wf.get("reasons") or diag.get("top_fail_reasons") or {}
    if isinstance(reasons, dict):
        reason_map = reasons
    else:
        reason_map = {str(r).strip().lower(): 1 for r in (reasons or []) if str(r).strip()}

    return {
        "trade_activity": activity,
        "top_fail_reasons": reason_map,
        "mean_test_trades": mean_test,
        "zero_test_folds": zero_folds,
        "passed": bool(wf.get("passed", False)),
        "score": _safe_float(wf.get("score", 0.0), 0.0),
    }


def _pick_parent(symbol: str, timeframe: str) -> dict[str, Any] | None:
    strategies = list_strategies(active_only=True)
    symbol_key = symbol.strip().lower()
    tf_key = timeframe.strip().lower()

    candidates = []
    for row in strategies:
        tags = {str(t).strip().lower() for t in (row.get("tags") or []) if str(t).strip()}
        if symbol_key in tags and tf_key in tags:
            candidates.append(row)

    if not candidates:
        return None

    def _rank(row: dict[str, Any]):
        metrics = row.get("metrics") or {}
        wf = metrics.get("walk_forward") or {}
        return (
            _safe_float(wf.get("score", 0.0)),
            _safe_float(metrics.get("profit_factor", 0.0)),
            _safe_float(metrics.get("return_pct", 0.0)),
        )

    return sorted(candidates, key=_rank, reverse=True)[0]


def _record_evolution_audit(child, parent, cycle_id, *, status, score=0.0, passed=False, metrics=None, notes=""):
    try:
        record_evolution_run(
            cycle_id=cycle_id,
            symbol=child.symbol,
            timeframe=child.timeframe,
            parent_strategy_id=parent.get("strategy_id") if parent else None,
            child_strategy_id=child.strategy_id,
            status=status,
            score=score,
            passed=passed,
            parameters=child.parameters,
            metrics=metrics or {},
            notes=notes,
        )
    except Exception as exc:
        print(f"[WARN] evolution audit failed: {exc}", flush=True)


def _run_split(child, start, end, allow_shorts, max_bars, use_cache):
    payload = {
        "strategy_id": child.base_strategy,
        "base_strategy": child.base_strategy,
        "version": child.version - 1,
        "parameters": child.parameters,
    }
    return run_backtest(
        child.symbol,
        child.timeframe,
        start=start,
        end=end,
        allow_shorts=allow_shorts or bool(child.parameters.get("allow_shorts", False)),
        max_bars=max_bars,
        use_cache=use_cache,
        strategy_override=payload,
    )


def evolve_once(
    *,
    symbols: list[str],
    timeframes: list[str],
    children_per_parent: int = 4,
    max_bars: int = 0,
    allow_shorts: bool = False,
    start: str | None = None,
    end: str | None = None,
    lookback_days: int = 720,
    folds: int = 3,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    use_cache: bool = True,
    family: str = "evo",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    try:
        init_db()
    except Exception as exc:
        print(f"[WARN] DB init failed, continuing in local mode: {exc}", flush=True)

    cycle_id = f"{family}_{_now_iso().replace(':', '').replace('-', '')}"
    if not start or not end:
        start, end = default_evolution_window(lookback_days)

    results: list[dict[str, Any]] = []

    for symbol in symbols:
        for timeframe in timeframes:
            parent = _pick_parent(symbol, timeframe)
            parent_feedback = _parent_feedback_from_metrics(parent.get("metrics") if parent else None)
            children = (
                [seed_strategy(symbol, timeframe, family=family)]
                if parent is None
                else mutate_parent(
                    parent,
                    symbol=symbol,
                    timeframe=timeframe,
                    n_children=children_per_parent,
                    seed=seed,
                    feedback=parent_feedback,
                )
            )

            wf_folds = build_walk_forward_folds(
                start,
                end,
                folds=folds,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

            for child in children:
                upsert_strategy(
                    child.strategy_id,
                    base_strategy=child.base_strategy,
                    version=child.version,
                    status="candidate",
                    parameters=child.parameters,
                    metrics={},
                    tags=child.tags,
                    source=child.source,
                    notes=child.notes,
                    active=False,
                )
                _record_evolution_audit(child, parent, cycle_id, status="running", notes="walk-forward eval")

                fold_reports = []
                for fold in wf_folds:
                    start_ts = datetime.fromisoformat(fold.start.replace("Z", "+00:00"))
                    end_ts = datetime.fromisoformat(fold.end.replace("Z", "+00:00"))
                    span = end_ts - start_ts
                    train_end = start_ts + span * train_ratio
                    val_end = train_end + span * val_ratio

                    train = _run_split(child, fold.start, train_end.isoformat(), allow_shorts, max_bars, use_cache)
                    val = _run_split(child, train_end.isoformat(), val_end.isoformat(), allow_shorts, max_bars, use_cache)
                    test = _run_split(child, val_end.isoformat(), fold.end, allow_shorts, max_bars, use_cache)

                    fold_reports.append({"label": fold.label, "train": train, "val": val, "test": test})

                summary = summarize_walk_forward_reports(fold_reports, timeframe=timeframe)
                metrics = {"walk_forward": summary, "folds": fold_reports}
                metrics["diagnostics"] = build_candidate_diagnostics({
                    "strategy_id": child.strategy_id,
                    "symbol": child.symbol,
                    "timeframe": child.timeframe,
                    "walk_forward": summary,
                })

                status = "validated" if summary["passed"] else "rejected"
                upsert_strategy(
                    child.strategy_id,
                    base_strategy=child.base_strategy,
                    version=child.version,
                    status=status,
                    parameters=child.parameters,
                    metrics=metrics,
                    tags=child.tags,
                    source=child.source,
                    notes=child.notes,
                    active=summary["passed"],
                    validated_at=_now_iso() if summary["passed"] else None,
                )

                record_experiment(
                    child.strategy_id,
                    symbol=child.symbol,
                    timeframe=child.timeframe,
                    run_type="walkforward_backtest",
                    parameters=child.parameters,
                    metrics=metrics,
                    passed=summary["passed"],
                    notes=f"cycle_id={cycle_id}",
                )

                _record_evolution_audit(
                    child,
                    parent,
                    cycle_id,
                    status=status,
                    score=summary["score"],
                    passed=summary["passed"],
                    metrics=metrics,
                    notes="promoted" if summary["passed"] else "rejected",
                )

                results.append({
                    "strategy_id": child.strategy_id,
                    "symbol": child.symbol,
                    "timeframe": child.timeframe,
                    "walk_forward": summary,
                    "diagnostics": metrics["diagnostics"],
                    "feedback_profile": parent_feedback,
                })

    return results


def continuous_evolution(**kwargs):
    cycle = 0
    while True:
        cycle += 1
        results = evolve_once(**kwargs)
        print(json.dumps({"cycle": cycle, "results": results}, indent=2), flush=True)
        if kwargs.get("max_cycles") and cycle >= kwargs["max_cycles"]:
            break
        time.sleep(max(1, kwargs.get("sleep_seconds", 3600)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--children-per-parent", type=int, default=4)
    parser.add_argument("--max-bars", type=int, default=0)
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--lookback-days", type=int, default=720)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--family", default="evo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=int, default=3600)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    if args.continuous:
        continuous_evolution(
            symbols=symbols,
            timeframes=timeframes,
            children_per_parent=args.children_per_parent,
            max_bars=args.max_bars,
            allow_shorts=args.allow_shorts,
            start=args.start,
            end=args.end,
            lookback_days=args.lookback_days,
            folds=args.folds,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            use_cache=not args.no_cache,
            family=args.family,
            seed=args.seed,
            sleep_seconds=args.sleep_seconds,
            max_cycles=None if args.max_cycles <= 0 else args.max_cycles,
        )
    else:
        results = evolve_once(
            symbols=symbols,
            timeframes=timeframes,
            children_per_parent=args.children_per_parent,
            max_bars=args.max_bars,
            allow_shorts=args.allow_shorts,
            start=args.start,
            end=args.end,
            lookback_days=args.lookback_days,
            folds=args.folds,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            use_cache=not args.no_cache,
            family=args.family,
            seed=args.seed,
        )
        print(json.dumps({"results": results}, indent=2))
