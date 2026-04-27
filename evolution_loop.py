"""Fully automated strategy evolution loop.

Now includes:
- walk-forward validation (train / validation / test splits)
- rolling folds for robustness
- stability-aware scoring (reduces overfitting)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

from backtest import run_backtest
from db import init_db
from mutation_engine import MutationSpec, mutate_parent, seed_strategy
from strategy_registry import (
    list_strategies,
    record_experiment,
    upsert_strategy,
    record_evolution_run,
)
from validation import (
    default_evolution_window,
    build_walk_forward_folds,
    summarize_walk_forward_reports,
)

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LINK/USDT", "AVAX/USDT", "SOL/USDT"]
DEFAULT_TIMEFRAMES = ["1d", "4h"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
        score = _safe_float(wf.get("score", 0.0))
        return (
            score,
            _safe_float(metrics.get("profit_factor", 0.0)),
            _safe_float(metrics.get("return_pct", 0.0)),
        )

    return sorted(candidates, key=_rank, reverse=True)[0]


def _insert_evolution_audit(child, parent, cycle_id, *, status, score=0.0, passed=False, metrics=None, notes=""):
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
    except Exception as e:
        print(f"[WARN] evolution audit failed: {e}", flush=True)


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
    except Exception as e:
        print(f"[WARN] DB init failed, using local mode: {e}", flush=True)

    cycle_id = f"{family}_{_now_iso().replace(':', '').replace('-', '')}"

    if not start or not end:
        start, end = default_evolution_window(lookback_days)

    results = []

    for symbol in symbols:
        for timeframe in timeframes:
            parent = _pick_parent(symbol, timeframe)
            children = (
                [seed_strategy(symbol, timeframe, family=family)]
                if parent is None
                else mutate_parent(parent, symbol=symbol, timeframe=timeframe, n_children=children_per_parent, seed=seed)
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

                _insert_evolution_audit(child, parent, cycle_id, status="running", notes="walk-forward eval")

                fold_reports = []

                for fold in wf_folds:
                    fold_start = fold.start
                    fold_end = fold.end

                    start_ts = datetime.fromisoformat(fold_start.replace("Z", "+00:00"))
                    end_ts = datetime.fromisoformat(fold_end.replace("Z", "+00:00"))
                    span = end_ts - start_ts

                    train_end = start_ts + span * train_ratio
                    val_end = train_end + span * val_ratio

                    train = _run_split(child, fold_start, train_end.isoformat(), allow_shorts, max_bars, use_cache)
                    val = _run_split(child, train_end.isoformat(), val_end.isoformat(), allow_shorts, max_bars, use_cache)
                    test = _run_split(child, val_end.isoformat(), fold_end, allow_shorts, max_bars, use_cache)

                    fold_reports.append({
                        "label": fold.label,
                        "train": train,
                        "val": val,
                        "test": test,
                    })

                summary = summarize_walk_forward_reports(fold_reports, timeframe=timeframe)

                status = "validated" if summary["passed"] else "rejected"

                metrics = {
                    "walk_forward": summary,
                    "folds": fold_reports,
                }

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

                _insert_evolution_audit(
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
            max_cycles=args.max_cycles,
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