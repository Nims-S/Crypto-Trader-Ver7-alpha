from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from backtest import run_backtest
from diagnostics import build_candidate_diagnostics
from db import init_db
from mutation_engine import mutate_parent, seed_strategy
from strategy_registry import list_strategies, record_experiment, record_evolution_run, upsert_strategy
from validation import build_walk_forward_folds, default_evolution_window, summarize_walk_forward_reports

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LINK/USDT", "AVAX/USDT", "SOL/USDT"]
DEFAULT_TIMEFRAMES = ["1d", "4h"]


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return d


def _pick_parent(symbol, timeframe):
    rows = list_strategies(active_only=True)
    s = symbol.lower(); t = timeframe.lower()
    matches = [r for r in rows if s in {str(x).lower() for x in (r.get("tags") or [])} and t in {str(x).lower() for x in (r.get("tags") or [])}]
    if not matches:
        return None
    return sorted(matches, key=lambda r: _safe_float(((r.get("metrics") or {}).get("walk_forward") or {}).get("score", 0.0)), reverse=True)[0]


def _feedback_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    wf = metrics.get("walk_forward") or {}
    diag = metrics.get("diagnostics") or {}

    activity = diag.get("trade_activity") or {}
    mean_test = 0.0
    if isinstance(activity, dict):
        mean_bucket = activity.get("mean") or {}
        if isinstance(mean_bucket, dict):
            mean_test = _safe_float(mean_bucket.get("test", 0), 0.0)

    return {
        "top_fail_reasons": wf.get("reasons") or diag.get("top_fail_reasons") or {},
        "trade_activity": activity,
        "mean_test_trades": mean_test,
        "passed": bool(wf.get("passed", False)),
        "score": _safe_float(wf.get("score", 0.0), 0.0),
    }


def _too_restrictive(params: dict[str, Any]) -> bool:
    flags = [
        params.get("use_htf_filter", False),
        params.get("use_volume_filter", False),
        params.get("use_structure_filter", False),
        params.get("use_trend_filter", False),
    ]
    high_thresholds = (
        _safe_float(params.get("min_adx", 0), 0) > 10
        and _safe_float(params.get("min_atr_rank", 0), 0) > 0.08
        and _safe_float(params.get("min_bb_rank", 0), 0) > 0.08
    )
    return sum(1 for f in flags if f) >= 3 and high_thresholds


def _run_split(child, start, end, allow_shorts, max_bars, use_cache):
    override = {"strategy_id": child.base_strategy, "base_strategy": child.base_strategy, "version": child.version - 1, "parameters": child.parameters}
    return run_backtest(child.symbol, child.timeframe, start=start, end=end, allow_shorts=allow_shorts or bool(child.parameters.get("allow_shorts", False)), max_bars=max_bars, use_cache=use_cache, strategy_override=override)


def evolve_once(symbols, timeframes, children_per_parent=4, max_bars=0, allow_shorts=False, start=None, end=None, lookback_days=720, folds=3, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, use_cache=True, family="evo", seed=None):
    try:
        init_db()
    except Exception:
        print("[WARN] DB init skipped (local mode or psycopg2 unavailable)")

    if not start or not end:
        start, end = default_evolution_window(lookback_days)

    cycle_id = f"{family}_{_now_iso().replace(':','').replace('-','')}"
    results = []

    for symbol in symbols:
        for timeframe in timeframes:
            parent = _pick_parent(symbol, timeframe)
            parent_feedback = _feedback_from_metrics((parent or {}).get("metrics") or {})

            children = [seed_strategy(symbol, timeframe, family=family)] if parent is None else mutate_parent(parent, symbol=symbol, timeframe=timeframe, n_children=children_per_parent, seed=seed, feedback=parent_feedback)

            wf_folds = build_walk_forward_folds(start, end, folds=folds, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

            # Anti-starvation safety: keep at least one child per symbol/timeframe.
            candidate_rows = []
            for child in children:
                is_restrictive = _too_restrictive(child.parameters)
                should_skip = parent is not None and parent_feedback.get("mean_test_trades", 0) < 3 and is_restrictive
                candidate_rows.append((child, should_skip, is_restrictive))

            if candidate_rows and all(skip for _, skip, _ in candidate_rows):
                # Force the least restrictive child to survive so the generation is never empty.
                candidate_rows.sort(
                    key=lambda item: (
                        item[2],
                        _safe_float(item[0].parameters.get("min_adx", 0), 0.0),
                        _safe_float(item[0].parameters.get("min_bb_rank", 0), 0.0),
                        _safe_float(item[0].parameters.get("min_atr_rank", 0), 0.0),
                    )
                )
                candidate_rows[0] = (candidate_rows[0][0], False, candidate_rows[0][2])

            for child, should_skip, _ in candidate_rows:
                if should_skip:
                    continue

                upsert_strategy(child.strategy_id, base_strategy=child.base_strategy, version=child.version, status="candidate", parameters=child.parameters, metrics={}, tags=child.tags, source=child.source, notes=child.notes, active=False)

                try:
                    record_evolution_run(cycle_id=cycle_id, symbol=child.symbol, timeframe=child.timeframe, parent_strategy_id=(parent or {}).get("strategy_id"), child_strategy_id=child.strategy_id, status="running")
                except Exception:
                    pass

                fold_reports = []
                for fold in wf_folds:
                    st = datetime.fromisoformat(fold.start.replace("Z", "+00:00"))
                    en = datetime.fromisoformat(fold.end.replace("Z", "+00:00"))
                    span = en - st
                    train_end = st + span * train_ratio
                    val_end = train_end + span * val_ratio

                    train = _run_split(child, fold.start, train_end.isoformat(), allow_shorts, max_bars, use_cache)
                    val = _run_split(child, train_end.isoformat(), val_end.isoformat(), allow_shorts, max_bars, use_cache)
                    test = _run_split(child, val_end.isoformat(), fold.end, allow_shorts, max_bars, use_cache)

                    fold_reports.append({"train": train, "val": val, "test": test})

                summary = summarize_walk_forward_reports(fold_reports, timeframe=timeframe)
                diagnostics = build_candidate_diagnostics({"strategy_id": child.strategy_id, "symbol": child.symbol, "timeframe": child.timeframe, "walk_forward": summary})
                metrics = {"walk_forward": summary, "diagnostics": diagnostics}
                passed = bool(summary.get("passed"))

                upsert_strategy(child.strategy_id, base_strategy=child.base_strategy, version=child.version, status=("validated" if passed else "rejected"), parameters=child.parameters, metrics=metrics, tags=child.tags, source=child.source, notes=child.notes, active=passed, validated_at=_now_iso() if passed else None)

                record_experiment(child.strategy_id, symbol=child.symbol, timeframe=child.timeframe, run_type="walkforward_backtest", parameters=child.parameters, metrics=metrics, passed=passed, notes=f"cycle_id={cycle_id}")

                results.append({
                    "strategy_id": child.strategy_id,
                    "symbol": child.symbol,
                    "timeframe": child.timeframe,
                    "walk_forward": summary,
                    "feedback": parent_feedback,
                })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    print(json.dumps({"results": evolve_once(symbols, timeframes)}, indent=2))
