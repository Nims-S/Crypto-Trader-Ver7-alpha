"""Safe promotion pipeline for elevating winning strategies into the stable architecture layer.

Two operating modes are supported:
- strict: only validated, high-quality candidates are considered for promotion
- research: surfaces borderline candidates for review without auto-promoting them

The module writes immutable catalog/report snapshots and can optionally mark
strategies as architecture_promoted when the strict gate is satisfied.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strategy_registry import list_strategies, upsert_strategy


CATALOG_PATH = Path(os.getenv("ARCHITECTURE_CATALOG_FILE", "architecture_catalog.json"))
REPORT_PATH = Path(os.getenv("ARCHITECTURE_PROMOTION_REPORT_FILE", "architecture_promotion_report.json"))


@dataclass(frozen=True)
class PromotionPolicy:
    min_score: float = 0.55
    min_robustness_score: float = 0.40
    min_trades: int = 20
    min_profit_factor: float = 1.05
    min_win_rate: float = 0.40
    max_drawdown_pct: float = -20.0
    require_active: bool = True
    require_validated: bool = True
    max_per_symbol: int = 3
    max_total: int = 10
    allowed_regimes: tuple[str, ...] = ("trend", "mean_reversion", "breakout")


STRICT_POLICY = PromotionPolicy()
RESEARCH_POLICY = PromotionPolicy(
    min_score=0.20,
    min_robustness_score=0.05,
    min_trades=1,
    min_profit_factor=0.90,
    min_win_rate=0.20,
    max_drawdown_pct=-35.0,
    require_active=False,
    require_validated=False,
    max_per_symbol=5,
    max_total=15,
)


@dataclass(frozen=True)
class PromotionResult:
    strategy_id: str
    symbol: str | None
    timeframe: str | None
    status: str
    reason: str
    score: float
    robustness_score: float
    trades: int
    profit_factor: float
    win_rate: float
    max_drawdown_pct: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _load_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    decision = metrics.get("decision") or {}
    wf = metrics.get("walk_forward") or {}
    merged = dict(decision)
    if "score" not in merged and isinstance(wf, dict):
        merged["score"] = wf.get("score", 0.0)
    if "passed" not in merged and isinstance(wf, dict):
        merged["passed"] = wf.get("passed", False)
    if "reasons" not in merged and isinstance(wf, dict):
        merged["reasons"] = wf.get("reasons", [])
    return merged


def _latest_metric_score(metrics: dict[str, Any]) -> float:
    return _safe_float(_load_decision(metrics).get("score", 0.0), 0.0)


def _latest_passed(metrics: dict[str, Any]) -> bool:
    return _safe_bool(_load_decision(metrics).get("passed", False), False)


def _latest_reason(metrics: dict[str, Any]) -> str:
    reasons = _load_decision(metrics).get("reasons") or []
    if not reasons:
        return "passed"
    if isinstance(reasons, (list, tuple)):
        return ";".join(str(r) for r in reasons[:5])
    return str(reasons)


def _symbol_from_tags(tags: list[str]) -> str | None:
    for tag in tags:
        t = str(tag).strip().upper()
        if "/USDT" in t:
            return t
    return None


def _timeframe_from_tags(tags: list[str]) -> str | None:
    known = {"1d", "12h", "8h", "4h", "2h", "1h", "30m", "15m"}
    for tag in tags:
        t = str(tag).strip().lower()
        if t in known:
            return t
    return None


def _latest_trades(metrics: dict[str, Any]) -> int:
    wf = metrics.get("walk_forward") or {}
    split_results = wf.get("split_results") or {}
    best = 0
    for split_name in ("train", "val", "test"):
        for row in split_results.get(split_name) or []:
            best = max(best, _safe_int(row.get("trades", 0), 0))
    return best if best else _safe_int(metrics.get("trades", 0), 0)


def _latest_profit_factor(metrics: dict[str, Any]) -> float:
    wf = metrics.get("walk_forward") or {}
    split_results = wf.get("split_results") or {}
    best = 0.0
    for split_name in ("train", "val", "test"):
        for row in split_results.get(split_name) or []:
            best = max(best, _safe_float(row.get("profit_factor", 0.0), 0.0))
    return best if best else _safe_float(metrics.get("profit_factor", 0.0), 0.0)


def _latest_win_rate(metrics: dict[str, Any]) -> float:
    wf = metrics.get("walk_forward") or {}
    split_results = wf.get("split_results") or {}
    best = 0.0
    for split_name in ("train", "val", "test"):
        for row in split_results.get(split_name) or []:
            best = max(best, _safe_float(row.get("win_rate", 0.0), 0.0))
    return best if best else _safe_float(metrics.get("win_rate", 0.0), 0.0)


def _latest_drawdown(metrics: dict[str, Any]) -> float:
    wf = metrics.get("walk_forward") or {}
    split_results = wf.get("split_results") or {}
    worst = 0.0
    for split_name in ("train", "val", "test"):
        for row in split_results.get(split_name) or []:
            dd = _safe_float(row.get("max_drawdown_pct", 0.0), 0.0)
            if dd < worst:
                worst = dd
    return worst if worst else _safe_float(metrics.get("max_drawdown_pct", 0.0), 0.0)


def _eligible(row: dict[str, Any], policy: PromotionPolicy) -> tuple[bool, str, dict[str, Any]]:
    metrics = row.get("metrics") or {}
    tags = [str(t) for t in (row.get("tags") or [])]
    regime = str(row.get("regime_profile") or "").strip().lower()

    if policy.require_active and not _safe_bool(row.get("active", False), False):
        return False, "inactive", metrics
    if policy.require_validated and not _latest_passed(metrics):
        return False, "not_validated", metrics
    if regime and regime not in policy.allowed_regimes:
        return False, f"regime_blocked:{regime}", metrics

    score = _latest_metric_score(metrics)
    robustness = _safe_float(row.get("robustness_score", 0.0), 0.0)
    trades = _latest_trades(metrics)
    pf = _latest_profit_factor(metrics)
    wr = _latest_win_rate(metrics)
    dd = _latest_drawdown(metrics)

    if score < policy.min_score:
        return False, f"score<{policy.min_score:.2f}", metrics
    if robustness < policy.min_robustness_score:
        return False, f"robustness<{policy.min_robustness_score:.2f}", metrics
    if trades < policy.min_trades:
        return False, f"trades<{policy.min_trades}", metrics
    if pf < policy.min_profit_factor:
        return False, f"pf<{policy.min_profit_factor:.2f}", metrics
    if wr < policy.min_win_rate:
        return False, f"wr<{policy.min_win_rate:.2f}", metrics
    if dd <= policy.max_drawdown_pct:
        return False, f"dd<={policy.max_drawdown_pct:.1f}", metrics

    symbol = _symbol_from_tags(tags) or row.get("symbol")
    timeframe = _timeframe_from_tags(tags) or row.get("timeframe")
    payload = {
        "logic_hash": row.get("logic_hash"),
        "status": row.get("status"),
        "score": round(score, 6),
        "robustness_score": round(robustness, 6),
        "trades": trades,
        "profit_factor": round(pf, 4),
        "win_rate": round(wr, 4),
        "max_drawdown_pct": round(dd, 4),
        "reason": _latest_reason(metrics),
    }
    return True, "eligible", {**row, "symbol": symbol, "timeframe": timeframe, "promotion_payload": payload}


def _candidate_pool(policy: PromotionPolicy, symbol: str | None, timeframe: str | None, regime: str | None) -> list[dict[str, Any]]:
    rows = list_strategies(active_only=False)
    pool: list[dict[str, Any]] = []
    for row in rows:
        tags = [str(t).lower() for t in (row.get("tags") or [])]
        if symbol and symbol.lower() not in tags and symbol.upper() != row.get("symbol"):
            continue
        if timeframe and timeframe.lower() not in tags and timeframe.lower() != str(row.get("timeframe") or "").lower():
            continue
        if regime and str(row.get("regime_profile") or "").lower() != regime.lower():
            continue
        ok, reason, payload = _eligible(row, policy)
        if ok:
            pool.append(payload)
        elif policy is RESEARCH_POLICY:
            # In research mode, keep near-misses for review even if they miss thresholds.
            miss = dict(row)
            miss["promotion_reject_reason"] = reason
            miss["promotion_payload"] = {
                "logic_hash": row.get("logic_hash"),
                "status": row.get("status"),
                "score": round(_latest_metric_score(row.get("metrics") or {}), 6),
                "robustness_score": round(_safe_float(row.get("robustness_score", 0.0), 0.0), 6),
                "trades": _latest_trades(row.get("metrics") or {}),
                "profit_factor": round(_latest_profit_factor(row.get("metrics") or {}), 4),
                "win_rate": round(_latest_win_rate(row.get("metrics") or {}), 4),
                "max_drawdown_pct": round(_latest_drawdown(row.get("metrics") or {}), 4),
                "reason": reason,
                "review_only": True,
            }
            pool.append(miss)
    return pool


def select_promotion_candidates(
    *,
    policy: PromotionPolicy | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    regime: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    policy = policy or STRICT_POLICY
    pool = _candidate_pool(policy, symbol, timeframe, regime)

    def _rank(row: dict[str, Any]):
        metrics = row.get("metrics") or {}
        decision = _load_decision(metrics)
        return (
            _safe_float(decision.get("score", 0.0), 0.0),
            _safe_float(row.get("robustness_score", 0.0), 0.0),
            _safe_int(_candidate_trades(row), 0),
            row.get("updated_at") or row.get("created_at") or "",
        )

    pool.sort(key=_rank, reverse=True)
    return pool[: max(1, int(limit))]


def _candidate_trades(row: dict[str, Any]) -> int:
    payload = row.get("promotion_payload") or {}
    return _safe_int(payload.get("trades", 0), 0)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def promote_winners(
    *,
    policy: PromotionPolicy | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    regime: str | None = None,
    limit: int = 10,
    dry_run: bool = True,
) -> dict[str, Any]:
    policy = policy or STRICT_POLICY
    candidates = select_promotion_candidates(policy=policy, symbol=symbol, timeframe=timeframe, regime=regime, limit=limit)

    promoted: list[PromotionResult] = []
    for row in candidates:
        payload = row.get("promotion_payload") or {}
        strategy_id = row.get("strategy_id")
        is_review_only = bool(payload.get("review_only", False))
        is_strict_eligible = not is_review_only and _eligible(row, STRICT_POLICY)[0]

        promotion_result = PromotionResult(
            strategy_id=strategy_id,
            symbol=row.get("symbol") or symbol,
            timeframe=row.get("timeframe") or timeframe,
            status="architecture_promoted" if is_strict_eligible else "architecture_review",
            reason=payload.get("reason", "eligible"),
            score=_safe_float(payload.get("score", 0.0), 0.0),
            robustness_score=_safe_float(payload.get("robustness_score", 0.0), 0.0),
            trades=_safe_int(payload.get("trades", 0), 0),
            profit_factor=_safe_float(payload.get("profit_factor", 0.0), 0.0),
            win_rate=_safe_float(payload.get("win_rate", 0.0), 0.0),
            max_drawdown_pct=_safe_float(payload.get("max_drawdown_pct", 0.0), 0.0),
        )
        promoted.append(promotion_result)

        if not dry_run and is_strict_eligible:
            upsert_strategy(
                strategy_id,
                base_strategy=row.get("base_strategy") or strategy_id,
                version=int(row.get("version", 1) or 1),
                status="architecture_promoted",
                parameters=row.get("parameters") or {},
                metrics=row.get("metrics") or {},
                tags=list(row.get("tags") or []),
                source="architecture_promotion",
                notes=f"promoted_at={_now_iso()}",
                active=True,
                validated_at=_now_iso(),
                regime_profile=row.get("regime_profile") or "architecture",
                robustness_score=_safe_float(row.get("robustness_score", 0.0), 0.0),
                parent_strategy_id=row.get("parent_strategy_id"),
            )

    catalog = {
        "generated_at": _now_iso(),
        "dry_run": dry_run,
        "policy": asdict(policy),
        "count": len(promoted),
        "strategies": [asdict(item) for item in promoted],
    }
    report = {
        "generated_at": _now_iso(),
        "dry_run": dry_run,
        "policy": asdict(policy),
        "selected": len(candidates),
        "promoted": [asdict(item) for item in promoted],
    }

    _atomic_write(CATALOG_PATH, catalog)
    _atomic_write(REPORT_PATH, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote validated strategies into the architecture catalog safely")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--regime", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="Actually mark strict-eligible strategies as architecture_promoted in the registry")
    parser.add_argument("--mode", choices=["strict", "research"], default=os.getenv("ARCHITECTURE_PROMOTION_MODE", "research"))
    args = parser.parse_args()

    policy = STRICT_POLICY if args.mode == "strict" else RESEARCH_POLICY
    result = promote_winners(
        policy=policy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        regime=args.regime,
        limit=args.limit,
        dry_run=not args.apply,
    )
    print(json.dumps(result, indent=2))
