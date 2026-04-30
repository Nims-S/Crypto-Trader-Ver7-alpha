from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

from backtest import run_backtest
from diagnostics import build_candidate_diagnostics
from db import init_db
from feedback_engine import build_feedback_summary
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


def _split_mean(wf: dict[str, Any], split_name: str, key: str) -> float:
    split_results = wf.get("split_results") or {}
    rows = split_results.get(split_name) or []
    vals = []
    for row in rows:
        if isinstance(row, dict) and row.get(key) is not None:
            vals.append(_safe_float(row.get(key), 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def _score_row(r: dict[str, Any]) -> float:
    wf = (r.get("metrics") or {}).get("walk_forward") or {}
    score = _safe_float(wf.get("score", 0.0), 0.0)
    pf = _split_mean(wf, "test", "profit_factor")
    wr = _split_mean(wf, "test", "win_rate")
    trades = _split_mean(wf, "test", "trades")
    spread = _safe_float(wf.get("score_spread", 0.0), 0.0)
    bonus = 0.06 * min(pf, 2.0) + 0.04 * wr + 0.02 * min(trades / 20.0, 1.0) - 0.05 * min(spread, 0.5)
    return score + bonus


def _pick_parent(symbol, timeframe):
    s = symbol.lower()
    t = timeframe.lower()

    rows = list_strategies(active_only=True)
    matches = [r for r in rows if s in {str(x).lower() for x in (r.get("tags") or [])} and t in {str(x).lower() for x in (r.get("tags") or [])}]
    if matches:
        return sorted(matches, key=_score_row, reverse=True)[0]

    rows = list_strategies(active_only=False)
    matches = [r for r in rows if s in {str(x).lower() for x in (r.get("tags") or [])} and t in {str(x).lower() for x in (r.get("tags") or [])} and r.get("status") != "running"]

    if matches:
        return sorted(matches, key=_score_row, reverse=True)[0]

    return None


def _feedback_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    wf = metrics.get("walk_forward") or {}
    diag = metrics.get("diagnostics") or {}

    activity = diag.get("trade_activity") or {}
    means = (wf.get("means") or {}) if isinstance(wf.get("means"), dict) else {}

    return {
        "top_fail_reasons": wf.get("reasons") or diag.get("top_fail_reasons") or {},
        "trade_activity": activity,
        "mean_test_trades": _safe_float((activity.get("mean") or {}).get("test", 0), 0.0),
        "mean_val_trades": _safe_float((activity.get("mean") or {}).get("val", 0), 0.0),
        "mean_train_trades": _safe_float((activity.get("mean") or {}).get("train", 0), 0.0),
        "mean_test_pf": _safe_float((activity.get("mean_pf") or {}).get("test", 0), 0.0),
        "mean_val_pf": _safe_float((activity.get("mean_pf") or {}).get("val", 0), 0.0),
        "mean_train_pf": _safe_float((activity.get("mean_pf") or {}).get("train", 0), 0.0),
        "mean_test_wr": _safe_float((activity.get("mean_wr") or {}).get("test", 0), 0.0),
        "mean_val_wr": _safe_float((activity.get("mean_wr") or {}).get("val", 0), 0.0),
        "mean_train_wr": _safe_float((activity.get("mean_wr") or {}).get("train", 0), 0.0),
        "score": _safe_float(wf.get("score", 0.0), 0.0),
        "score_spread": _safe_float(wf.get("score_spread", 0.0), 0.0),
        "train_mean_score": _safe_float(means.get("train", 0.0), 0.0),
        "val_mean_score": _safe_float(means.get("val", 0.0), 0.0),
        "test_mean_score": _safe_float(means.get("test", 0.0), 0.0),
    }


def _merge_feedback(parent_feedback: dict, store_feedback: dict) -> dict:
    merged = dict(parent_feedback or {})
    for key, value in (store_feedback or {}).items():
        if key not in merged or not merged.get(key):
            merged[key] = value
    return merged


def _too_restrictive(params: dict[str, Any]) -> bool:
    flags = [params.get("use_htf_filter", False), params.get("use_volume_filter", False), params.get("use_structure_filter", False), params.get("use_trend_filter", False)]
    high_thresholds = (_safe_float(params.get("min_adx", 0), 0) > 10 and _safe_float(params.get("min_atr_rank", 0), 0) > 0.08 and _safe_float(params.get("min_bb_rank", 0), 0) > 0.08)
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
            store_feedback = build_feedback_summary(strategy_id=(parent or {}).get("strategy_id"), symbol=symbol, timeframe=timeframe)
            combined_feedback = _merge_feedback(parent_feedback, store_feedback)

            if parent is None:
                children = mutate_parent(None, symbol=symbol, timeframe=timeframe, n_children=children_per_parent, feedback=combined_feedback)
            else:
                children = mutate_parent(parent, symbol=symbol, timeframe=timeframe, n_children=children_per_parent, seed=seed, feedback=combined_feedback)

            wf_folds = build_walk_forward_folds(start, end, folds=folds, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

            candidate_rows = []
            for child in children:
                is_restrictive = _too_restrictive(child.parameters)
                should_skip = parent is not None and combined_feedback.get("mean_test_trades", 0) < 3 and is_restrictive
                candidate_rows.append((child, should_skip, is_restrictive))

            if candidate_rows and all(skip for _, skip, _ in candidate_rows):
                candidate_rows.sort(key=lambda item: (item[2], _safe_float(item[0].parameters.get("min_adx", 0), 0.0), _safe_float(item[0].parameters.get("min_bb_rank", 0), 0.0), _safe_float(item[0].parameters.get("min_atr_rank", 0), 0.0)))
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

                results.append({"strategy_id": child.strategy_id, "symbol": child.symbol, "timeframe": child.timeframe, "walk_forward": summary, "feedback": combined_feedback})

    return results


def evolve(symbols, timeframes, max_cycles=1, sleep_seconds=0, **kwargs):
    all_results = []
    cycles = max(1, int(max_cycles or 1))
    for cycle_idx in range(cycles):
        all_results.extend(evolve_once(symbols, timeframes, **kwargs))
        if cycle_idx < cycles - 1 and sleep_seconds:
            time.sleep(max(0, int(sleep_seconds)))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=int, default=0)
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
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    results = evolve(symbols, timeframes, max_cycles=args.max_cycles, sleep_seconds=args.sleep_seconds, children_per_parent=args.children_per_parent, max_bars=args.max_bars, allow_shorts=args.allow_shorts, start=args.start, end=args.end, lookback_days=args.lookback_days, folds=args.folds, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, use_cache=not args.no_cache, family=args.family, seed=args.seed)

    print(json.dumps({"results": results}, indent=2))
