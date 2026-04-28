"""Diagnostics utilities for understanding why candidates fail.

This is intentionally lightweight and safe to call inside evolution loops.
"""

from __future__ import annotations

from typing import Any


def summarize_walk_forward(report: dict[str, Any]) -> dict[str, Any]:
    wf = report.get("walk_forward", {})
    return {
        "score": wf.get("score"),
        "passed": wf.get("passed"),
        "reasons": wf.get("reasons", []),
        "means": wf.get("means", {}),
        "density_mean": wf.get("density_mean"),
        "score_spread": wf.get("score_spread"),
    }


def compact_reason_string(report: dict[str, Any]) -> str:
    wf = report.get("walk_forward", {})
    reasons = wf.get("reasons", [])
    if not reasons:
        return "passed"
    # Collapse duplicates like fold_1:train:pf<1.10 -> pf<1.10
    short = sorted({r.split(":")[-1] for r in reasons})
    return ",".join(short)
