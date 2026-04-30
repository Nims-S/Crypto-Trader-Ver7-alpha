"""Validation helpers for the automated evolution loop.

This module keeps date-window generation and walk-forward aggregation separate
from the orchestration layer so the evolution loop stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from evolution import score_metrics


@dataclass(frozen=True)
class WalkForwardSplit:
    label: str
    start: str
    end: str

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label, "start": self.start, "end": self.end}


TRADE_DENSITY_BASE = {
    "1d": 4,
    "12h": 5,
    "8h": 6,
    "4h": 6,
    "2h": 8,
    "1h": 10,
    "30m": 12,
    "15m": 14,
}

SOFT_DENSITY_FLOOR = 0.30


def _to_utc_timestamp(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        raise ValueError("timestamp value is required")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").isoformat() if ts.tzinfo else ts.tz_localize("UTC").isoformat()


def default_evolution_window(lookback_days: int = 720) -> tuple[str, str]:
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=max(30, int(lookback_days or 720)))
    return _iso(start), _iso(end)


def build_walk_forward_folds(
    start: str,
    end: str,
    *,
    folds: int = 3,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> list[WalkForwardSplit]:
    if folds <= 0:
        raise ValueError("folds must be positive")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("split ratios must be positive")

    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    span = end_ts - start_ts
    if span < pd.Timedelta(days=90):
        raise ValueError("walk-forward range is too short; use at least 90 days")

    train_len = span * train_ratio
    val_len = span * val_ratio
    test_len = span * test_ratio
    step = test_len

    folds_out: list[WalkForwardSplit] = []
    for fold_idx in range(folds):
        fold_start = start_ts + (step * fold_idx)
        train_end = fold_start + train_len
        val_end = train_end + val_len
        test_end = val_end + test_len

        if test_end > end_ts:
            break

        folds_out.append(WalkForwardSplit(label=f"fold_{fold_idx + 1}", start=_iso(fold_start), end=_iso(test_end)))

    if not folds_out:
        folds_out.append(WalkForwardSplit(label="fold_1", start=_iso(start_ts), end=_iso(end_ts)))

    return folds_out


def _trade_density_threshold(timeframe: str) -> int:
    tf = (timeframe or "").strip().lower()
    return TRADE_DENSITY_BASE.get(tf, 6)


def _trade_density_score(trades: int, timeframe: str, split_name: str) -> float:
    base = _trade_density_threshold(timeframe)
    if split_name == "train":
        target = max(2, int(round(base * 0.75)))
    else:
        target = max(2, int(round(base * 0.60)))
    return min(1.0, max(0.0, trades / float(target)))


def _decision_to_dict(decision: Any) -> dict[str, Any]:
    if isinstance(decision, dict):
        return decision
    try:
        return asdict(decision)
    except Exception:
        pass
    payload: dict[str, Any] = {}
    for key in ("score", "passed", "reasons"):
        if hasattr(decision, key):
            payload[key] = getattr(decision, key)
    return payload


def summarize_walk_forward_reports(
    fold_reports: list[dict[str, Any]],
    *,
    timeframe: str,
) -> dict[str, Any]:
    if not fold_reports:
        return {
            "score": 0.0,
            "passed": False,
            "reason": "no fold reports",
            "fold_count": 0,
        }

    split_scores: dict[str, list[float]] = {"train": [], "val": [], "test": []}
    split_decisions: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_results: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    density_scores: dict[str, list[float]] = {"train": [], "val": [], "test": []}
    reasons: list[str] = []

    for fold in fold_reports:
        for split_name in ("train", "val", "test"):
            result = fold.get(split_name) or {}
            split_results[split_name].append(result)
            if "error" in result:
                reasons.append(f"{fold.get('label', 'fold')}:{split_name}:{result['error']}")
                continue

            decision = score_metrics(result)
            split_scores[split_name].append(decision.score)
            split_decisions[split_name].append(_decision_to_dict(decision))

            trades = int(result.get("trades", 0) or 0)
            density = _trade_density_score(trades, timeframe, split_name)
            density_scores[split_name].append(density)

            if density < SOFT_DENSITY_FLOOR:
                reasons.append(f"{fold.get('label', 'fold')}:{split_name}:density<{SOFT_DENSITY_FLOOR:.2f}")

            if not decision.passed:
                reasons.extend([f"{fold.get('label', 'fold')}:{split_name}:{reason}" for reason in decision.reasons])

    train_mean = float(np.mean(split_scores["train"])) if split_scores["train"] else 0.0
    val_mean = float(np.mean(split_scores["val"])) if split_scores["val"] else 0.0
    test_mean = float(np.mean(split_scores["test"])) if split_scores["test"] else 0.0
    val_std = float(np.std(split_scores["val"])) if len(split_scores["val"]) > 1 else 0.0
    test_std = float(np.std(split_scores["test"])) if len(split_scores["test"]) > 1 else 0.0
    combined_scores = split_scores["train"] + split_scores["val"] + split_scores["test"]
    score_spread = float(max(combined_scores) - min(combined_scores)) if len(combined_scores) >= 2 else 0.0

    density_train = float(np.mean(density_scores["train"])) if density_scores["train"] else 0.0
    density_val = float(np.mean(density_scores["val"])) if density_scores["val"] else 0.0
    density_test = float(np.mean(density_scores["test"])) if density_scores["test"] else 0.0
    density_mean = float(np.mean([density_train, density_val, density_test]))

    composite = (0.15 * train_mean) + (0.35 * val_mean) + (0.50 * test_mean)
    stability_penalty = min(0.25, (val_std + test_std) * 0.5 + max(0.0, score_spread - 0.25) * 0.5)
    density_bonus = 0.12 * density_mean
    final_score = max(0.0, composite - stability_penalty + density_bonus)

    min_val_score = min(split_scores["val"], default=0.0)
    min_test_score = min(split_scores["test"], default=0.0)

    passed = (
        bool(split_scores["val"])
        and bool(split_scores["test"])
        and val_mean >= 0.55
        and test_mean >= 0.55
        and min_val_score >= 0.45
        and min_test_score >= 0.45
        and final_score >= 0.55
        and score_spread <= 0.35
        and not reasons
    )

    return {
        "score": round(final_score, 6),
        "passed": passed,
        "reasons": reasons,
        "fold_count": len(fold_reports),
        "composite": round(composite, 6),
        "stability_penalty": round(stability_penalty, 6),
        "density_bonus": round(density_bonus, 6),
        "density_mean": round(density_mean, 6),
        "score_spread": round(score_spread, 6),
        "means": {
            "train": round(train_mean, 6),
            "val": round(val_mean, 6),
            "test": round(test_mean, 6),
        },
        "stddev": {
            "val": round(val_std, 6),
            "test": round(test_std, 6),
        },
        "density_scores": {
            "train": round(density_train, 6),
            "val": round(density_val, 6),
            "test": round(density_test, 6),
        },
        "split_scores": split_scores,
        "split_decisions": split_decisions,
        "split_results": split_results,
    }
