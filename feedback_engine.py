"""Feedback classification for the evolutionary search loop.

This module reads the strategy store / evolution snapshots and turns rejection
patterns into concrete mutation directives. It is deliberately conservative:
- sparse trade activity causes filters to loosen
- weak profit factor shifts entries toward trend/breakout structures
- unstable equity curves reduce holding time and position aggressiveness
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_STORE_PATH = Path(".strategy_store.json")


@dataclass(frozen=True)
class FailureProfile:
    primary: str
    counts: dict[str, int]
    notes: tuple[str, ...]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def load_strategy_store(path: str | Path = DEFAULT_STORE_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _reason_bucket(reason: str) -> str:
    r = (reason or "").lower()
    if any(x in r for x in ("trades<", "no trades", "sparse", "activity", "trade density")):
        return "no_trades"
    if any(x in r for x in ("pf<", "profit factor", "low_pf")):
        return "low_profit_factor"
    if any(x in r for x in ("wr<", "win rate", "low_wr")):
        return "low_win_rate"
    if any(x in r for x in ("dd<", "drawdown", "max_drawdown", "dd>")):
        return "high_drawdown"
    if any(x in r for x in ("score_spread", "std", "unstable", "variance")):
        return "unstable"
    return "other"


def _extract_reasons(obj: Any) -> list[str]:
    reasons: list[str] = []
    if obj is None:
        return reasons
    if isinstance(obj, str):
        reasons.append(obj)
        return reasons
    if isinstance(obj, dict):
        for value in obj.values():
            reasons.extend(_extract_reasons(value))
        return reasons
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            reasons.extend(_extract_reasons(item))
    return reasons


def _extract_runs(store: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("evolution_runs", "runs", "experiments", "records"):
        value = store.get(key)
        if isinstance(value, list):
            return [r for r in value if isinstance(r, dict)]
    return []


def summarize_store_feedback(
    *,
    store_path: str | Path = DEFAULT_STORE_PATH,
    strategy_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Summarize the latest failure patterns from the strategy store."""
    store = load_strategy_store(store_path)
    runs = _extract_runs(store)

    if strategy_id:
        runs = [r for r in runs if str(r.get("strategy_id") or r.get("child_strategy_id") or r.get("id") or "") == str(strategy_id)]
    if symbol:
        s = symbol.lower()
        runs = [r for r in runs if s in {str(t).lower() for t in (r.get("tags") or [])} or s == str(r.get("symbol") or "").lower()]
    if timeframe:
        t = timeframe.lower()
        runs = [r for r in runs if t in {str(tag).lower() for tag in (r.get("tags") or [])} or t == str(r.get("timeframe") or "").lower()]

    runs = runs[-max(1, int(limit)) :]

    counts = {
        "no_trades": 0,
        "low_profit_factor": 0,
        "low_win_rate": 0,
        "high_drawdown": 0,
        "unstable": 0,
        "other": 0,
    }
    notes: list[str] = []

    train_trades: list[float] = []
    val_trades: list[float] = []
    test_trades: list[float] = []
    train_pf: list[float] = []
    val_pf: list[float] = []
    test_pf: list[float] = []
    train_wr: list[float] = []
    val_wr: list[float] = []
    test_wr: list[float] = []
    score_spreads: list[float] = []

    for run in runs:
        metrics = run.get("metrics") or {}
        wf = metrics.get("walk_forward") or {}

        score_spreads.append(_safe_float(wf.get("score_spread", metrics.get("score_spread", 0.0)), 0.0))

        reasons = _extract_reasons(wf.get("reasons"))
        reasons.extend(_extract_reasons(wf.get("split_decisions")))
        reasons.extend(_extract_reasons(run.get("notes")))

        for reason in reasons:
            bucket = _reason_bucket(reason)
            counts[bucket] = counts.get(bucket, 0) + 1
            if reason and reason not in notes:
                notes.append(reason)

        split_results = wf.get("split_results") or {}
        for split_name, tr_list, pf_list, wr_list in (
            ("train", train_trades, train_pf, train_wr),
            ("val", val_trades, val_pf, val_wr),
            ("test", test_trades, test_pf, test_wr),
        ):
            for row in split_results.get(split_name) or []:
                if not isinstance(row, dict):
                    continue
                tr_list.append(_safe_float(row.get("trades", 0), 0.0))
                pf_list.append(_safe_float(row.get("profit_factor", 0.0), 0.0))
                wr_list.append(_safe_float(row.get("win_rate", 0.0), 0.0))

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    def _primary_bucket() -> str:
        ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return ranked[0][0] if ranked and ranked[0][1] > 0 else "other"

    primary = _primary_bucket()
    trade_density = {
        "train": _mean(train_trades),
        "val": _mean(val_trades),
        "test": _mean(test_trades),
    }
    profit_factor = {
        "train": _mean(train_pf),
        "val": _mean(val_pf),
        "test": _mean(test_pf),
    }
    win_rate = {
        "train": _mean(train_wr),
        "val": _mean(val_wr),
        "test": _mean(test_wr),
    }

    return {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "runs_seen": len(runs),
        "failure_profile": FailureProfile(primary=primary, counts=counts, notes=tuple(notes[:25])).__dict__,
        "trade_activity": {
            "mean": trade_density,
            "mean_pf": profit_factor,
            "mean_wr": win_rate,
        },
        "score_spread": _mean(score_spreads),
        "mean_train_trades": trade_density["train"],
        "mean_val_trades": trade_density["val"],
        "mean_test_trades": trade_density["test"],
        "mean_train_pf": profit_factor["train"],
        "mean_val_pf": profit_factor["val"],
        "mean_test_pf": profit_factor["test"],
        "mean_train_wr": win_rate["train"],
        "mean_val_wr": win_rate["val"],
        "mean_test_wr": win_rate["test"],
        "top_fail_reasons": notes[:25],
    }


def derive_mutation_directives(feedback: dict[str, Any]) -> dict[str, Any]:
    """Translate failure patterns into mutation directives."""
    feedback = feedback or {}
    profile = feedback.get("failure_profile") or {}
    primary = str(profile.get("primary") or "other")
    counts = profile.get("counts") or {}

    trade_mean = _safe_float((feedback.get("trade_activity") or {}).get("mean", {}).get("test", 0.0), 0.0)
    pf_mean = _safe_float((feedback.get("trade_activity") or {}).get("mean_pf", {}).get("test", 0.0), 0.0)
    wr_mean = _safe_float((feedback.get("trade_activity") or {}).get("mean_wr", {}).get("test", 0.0), 0.0)
    spread = _safe_float(feedback.get("score_spread", 0.0), 0.0)

    directives: dict[str, Any] = {
        "explore_more": True,
        "loosen_filters": False,
        "tighten_exits": False,
        "shorten_holding": False,
        "prefer_structure": True,
        "prefer_breakout": False,
        "prefer_trend_pullback": False,
        "size_multiplier": 1.0,
    }

    if primary == "no_trades" or trade_mean < 3 or counts.get("no_trades", 0) >= max(counts.get("other", 0), 1):
        directives.update(
            {
                "loosen_filters": True,
                "prefer_breakout": True,
                "prefer_trend_pullback": False,
                "prefer_structure": False,
                "use_htf_filter": False,
                "use_volume_filter": False,
                "use_reclaim_filter": False,
                "use_structure_filter": False,
                "use_trend_filter": False,
                "min_adx_delta": -5.0,
                "min_atr_rank_multiplier": 0.5,
                "min_bb_rank_multiplier": 0.5,
                "entry_mode": "breakout",
                "tp1_close_fraction": 0.20,
                "tp2_close_fraction": 0.30,
                "be_trigger_rr": 1.2,
                "max_bars_override": 18,
            }
        )
    elif primary == "low_profit_factor" or (trade_mean >= 3 and pf_mean < 1.10):
        directives.update(
            {
                "tighten_exits": True,
                "prefer_breakout": pf_mean < 1.0,
                "prefer_trend_pullback": pf_mean >= 1.0,
                "entry_mode": "trend_pullback" if pf_mean >= 1.0 else "breakout",
                "use_breakout_filter": True,
                "use_trend_filter": True,
                "use_structure_filter": True,
                "use_volume_filter": True,
                "tp1_close_fraction": 0.18,
                "tp2_close_fraction": 0.34,
                "tp3_close_fraction": 0.20,
                "be_trigger_rr": 1.8,
                "trail_atr_mult": 1.15,
                "trail_ema20": True,
                "max_bars_override": 24,
            }
        )
    elif primary == "high_drawdown":
        directives.update(
            {
                "shorten_holding": True,
                "prefer_structure": True,
                "size_multiplier": 0.80,
                "trail_atr_mult": 1.10,
                "be_trigger_rr": 1.5,
                "max_bars_override": 14,
                "use_structure_filter": True,
                "use_volume_filter": True,
            }
        )
    elif primary == "unstable" or spread > 0.25:
        directives.update(
            {
                "loosen_filters": True,
                "prefer_breakout": True,
                "use_htf_filter": False,
                "use_structure_filter": False,
                "use_reclaim_filter": False,
                "entry_mode": "breakout",
                "size_multiplier": 0.90 if wr_mean < 0.45 else 1.0,
                "max_bars_override": 18,
            }
        )

    if wr_mean < 0.40 and trade_mean >= 3:
        directives["entry_mode"] = "mean_reversion"
        directives["use_reclaim_filter"] = True
        directives["use_volume_filter"] = False
        directives["max_bars_override"] = min(int(directives.get("max_bars_override", 24)), 16)

    if pf_mean < 1.0 and trade_mean >= 3:
        directives["size_multiplier"] = min(_safe_float(directives.get("size_multiplier", 1.0), 1.0), 0.90)

    return directives


def build_feedback_summary(
    *,
    store_path: str | Path = DEFAULT_STORE_PATH,
    strategy_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    feedback = summarize_store_feedback(
        store_path=store_path,
        strategy_id=strategy_id,
        symbol=symbol,
        timeframe=timeframe,
    )
    feedback["mutation_directives"] = derive_mutation_directives(feedback)
    return feedback
