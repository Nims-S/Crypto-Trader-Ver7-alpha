"""Fully automated strategy evolution loop.

This module is the glue between:
- mutation_engine: generates bounded strategy variants
- backtest: evaluates each child on historical data
- evolution: scores results and decides promotion
- strategy_registry: persists candidates, experiments, and audit rows

It is intentionally conservative: only entry filters and a small state subset
are mutated. Stop logic, exits, and sizing stay under the existing backtest and
live execution rules.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

from backtest import run_backtest
from db import get_conn, init_db
from evolution import promotion_status, score_metrics
from mutation_engine import MutationSpec, mutate_parent, seed_strategy
from strategy_registry import get_strategy, list_strategies, record_experiment, upsert_strategy

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LINK/USDT", "AVAX/USDT", "SOL/USDT"]
DEFAULT_TIMEFRAMES = ["1d", "4h"]


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pick_parent(symbol: str, timeframe: str) -> dict[str, Any] | None:
    """Choose the strongest existing candidate for the symbol/timeframe pair."""
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
        decision = metrics.get("decision") or {}
        score = _safe_float(decision.get("score", 0.0))
        return (
            score,
            _safe_float(metrics.get("profit_factor", 0.0)),
            _safe_float(metrics.get("return_pct", 0.0)),
            _safe_int(metrics.get("trades", 0)),
        )

    return sorted(candidates, key=_rank, reverse=True)[0]


def _strategy_record_from_child(child: MutationSpec, parent: dict[str, Any] | None, cycle_id: str) -> dict[str, Any]:
    parent_id = parent.get("strategy_id") if parent else None
    return {
        "cycle_id": cycle_id,
        "parent_strategy_id": parent_id,
        "child_strategy_id": child.strategy_id,
        "symbol": child.symbol,
        "timeframe": child.timeframe,
        "parameters": child.parameters,
        "tags": child.tags,
    }


def _insert_evolution_audit(child: MutationSpec, parent: dict[str, Any] | None, cycle_id: str, *, status: str, score: float = 0.0, passed: bool = False, metrics: dict[str, Any] | None = None, notes: str = "") -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO evolution_runs (
            cycle_id, symbol, timeframe,
            parent_strategy_id, child_strategy_id,
            status, score, passed,
            parameters, metrics, notes, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, NOW())
        """,
        (
            cycle_id,
            child.symbol,
            child.timeframe,
            parent.get("strategy_id") if parent else None,
            child.strategy_id,
            status,
            score,
            passed,
            json.dumps(child.parameters, default=str),
            json.dumps(metrics or {}, default=str),
            notes,
        ),
    )
    conn.commit()
    conn.close()


def _evaluate_child(child: MutationSpec, *, start: str | None, end: str | None, max_bars: int, allow_shorts: bool, use_cache: bool) -> dict[str, Any]:
    parent_payload = {
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
        strategy_override=parent_payload,
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
    use_cache: bool = True,
    family: str = "evo",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run one deterministic evolution cycle across the requested universe."""
    init_db()
    cycle_id = f"{family}_{_now_iso().replace(':', '').replace('-', '')}"
    results: list[dict[str, Any]] = []

    for symbol in symbols:
        for timeframe in timeframes:
            parent = _pick_parent(symbol, timeframe)
            if parent is None:
                seed_child = seed_strategy(symbol, timeframe, family=family)
                children = [seed_child]
            else:
                children = mutate_parent(parent, symbol=symbol, timeframe=timeframe, n_children=children_per_parent, seed=seed)

            for child in children:
                # Pre-register the candidate before evaluation so the run is auditable.
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
                _insert_evolution_audit(child, parent, cycle_id, status="running", notes="candidate scheduled")

                result = _evaluate_child(
                    child,
                    start=start,
                    end=end,
                    max_bars=max_bars,
                    allow_shorts=allow_shorts,
                    use_cache=use_cache,
                )
                if "error" in result:
                    _insert_evolution_audit(child, parent, cycle_id, status="error", metrics=result, notes=result["error"])
                    results.append({"strategy_id": child.strategy_id, "status": "error", "error": result["error"]})
                    continue

                decision = score_metrics(result)
                status = promotion_status(decision)
                metrics = {**result, "decision": decision.as_dict()}

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
                    active=decision.passed,
                    validated_at=_now_iso() if decision.passed else None,
                )
                record_experiment(
                    child.strategy_id,
                    symbol=child.symbol,
                    timeframe=child.timeframe,
                    run_type="evolution_backtest",
                    parameters=child.parameters,
                    metrics=metrics,
                    passed=decision.passed,
                    notes=f"cycle_id={cycle_id}",
                )
                _insert_evolution_audit(child, parent, cycle_id, status=status, score=decision.score, passed=decision.passed, metrics=metrics, notes="promoted" if decision.passed else "rejected")
                results.append(
                    {
                        "strategy_id": child.strategy_id,
                        "parent_strategy_id": parent.get("strategy_id") if parent else None,
                        "symbol": child.symbol,
                        "timeframe": child.timeframe,
                        "decision": decision.as_dict(),
                        "result": result,
                    }
                )

    return results


def continuous_evolution(
    *,
    symbols: list[str],
    timeframes: list[str],
    children_per_parent: int = 4,
    max_bars: int = 0,
    allow_shorts: bool = False,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
    family: str = "evo",
    seed: int | None = None,
    sleep_seconds: int = 3600,
    max_cycles: int | None = None,
) -> None:
    cycle = 0
    while True:
        cycle += 1
        results = evolve_once(
            symbols=symbols,
            timeframes=timeframes,
            children_per_parent=children_per_parent,
            max_bars=max_bars,
            allow_shorts=allow_shorts,
            start=start,
            end=end,
            use_cache=use_cache,
            family=family,
            seed=seed,
        )
        print(json.dumps({"cycle": cycle, "results": results}, indent=2), flush=True)
        if max_cycles is not None and cycle >= max_cycles:
            break
        time.sleep(max(1, sleep_seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the automated strategy evolution loop")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--children-per-parent", type=int, default=4)
    parser.add_argument("--max-bars", type=int, default=0)
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
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
            use_cache=not args.no_cache,
            family=args.family,
            seed=args.seed,
        )
        print(json.dumps({"results": results}, indent=2))
