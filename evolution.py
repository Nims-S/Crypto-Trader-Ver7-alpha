"""Strategy scoring and promotion rules.

This module keeps the heuristics for deciding whether a strategy experiment is
worth promoting out of the backtest runner and live execution loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def score_metrics(
    metrics: dict[str, Any],
    *,
    min_trades: int = 0,
    min_profit_factor: float = 1.10,
    min_win_rate: float = 0.45,
    max_drawdown_pct: float = -15.0,
) -> ScoreDecision:
    """Return a normalized score plus a promotion decision.

    Trade count is handled as a soft factor now, not a hard gate. This allows
    sparse but potentially promising strategies to compete while still being
    penalized if they do not generate enough activity.
    """
    trades = int(metrics.get("trades", 0) or 0)
    profit_factor = _safe_float(metrics.get("profit_factor", 0.0))
    win_rate = _safe_float(metrics.get("win_rate", 0.0))
    drawdown = _safe_float(metrics.get("max_drawdown_pct", 0.0))
    return_pct = _safe_float(metrics.get("return_pct", 0.0))
    avg_rr = _safe_float(metrics.get("avg_rr_realised", 0.0))

    reasons: list[str] = []

    if min_trades > 0 and trades < min_trades:
        reasons.append(f"trades<{min_trades}")
    if profit_factor < min_profit_factor:
        reasons.append(f"pf<{min_profit_factor:.2f}")
    if win_rate < min_win_rate:
        reasons.append(f"wr<{min_win_rate:.2f}")
    if drawdown <= max_drawdown_pct:
        reasons.append(f"dd<={max_drawdown_pct:.1f}")

    # Soft score components: clipped so one outlier metric cannot dominate.
    pf_component = min(max((profit_factor - 1.0) / 1.5, 0.0), 1.0)
    wr_component = min(max((win_rate - 0.35) / 0.45, 0.0), 1.0)
    dd_component = min(max((abs(drawdown) - 5.0) / 20.0, 0.0), 1.0)
    ret_component = min(max(return_pct / 25.0, -1.0), 1.0)
    rr_component = min(max(avg_rr / 4.0, 0.0), 1.0)

    # Trade activity is now a soft scaling term. 20 trades is no longer a gate;
    # it is just the point where the activity bonus saturates.
    trade_target = max(6, min_trades if min_trades > 0 else 20)
    trade_component = min(max(trades / float(trade_target), 0.0), 2.0) / 2.0

    score = (
        0.26 * pf_component
        + 0.22 * wr_component
        + 0.18 * (1.0 - dd_component)
        + 0.12 * max(ret_component, 0.0)
        + 0.10 * rr_component
        + 0.12 * trade_component
    )

    passed = len(reasons) == 0 and score >= 0.55
    return ScoreDecision(score=score, passed=passed, reasons=tuple(reasons))


def promotion_status(decision: ScoreDecision) -> str:
    return "active" if decision.passed else "candidate"
