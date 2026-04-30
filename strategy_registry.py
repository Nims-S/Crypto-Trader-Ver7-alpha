"""Persistence helpers for strategy evolution (extended, backward compatible)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

_STORE_PATH = Path(os.getenv("STRATEGY_STORE_FILE", ".strategy_store.json"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_logic_hash(parameters: dict[str, Any] | None) -> str:
    try:
        blob = json.dumps(parameters or {}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _load() -> dict[str, Any]:
    if not _STORE_PATH.exists():
        return {"registry": {}, "experiments": [], "evolution_runs": [], "counters": {"experiment_id": 0, "evolution_id": 0}}
    try:
        with _STORE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        data = {}
    data.setdefault("registry", {})
    data.setdefault("experiments", [])
    data.setdefault("evolution_runs", [])
    data.setdefault("counters", {"experiment_id": 0, "evolution_id": 0})
    return data


def _save(store: dict[str, Any]) -> None:
    tmp = _STORE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, sort_keys=True, default=str)
    tmp.replace(_STORE_PATH)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _row(strategy_id: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "strategy_id": strategy_id,
        "base_strategy": row.get("base_strategy", "unknown"),
        "version": int(row.get("version", 1) or 1),
        "status": row.get("status", "candidate"),
        "parameters": row.get("parameters", {}) or {},
        "metrics": row.get("metrics", {}) or {},
        "tags": row.get("tags", []) or [],
        "source": row.get("source", "manual"),
        "notes": row.get("notes", "") or "",
        "active": bool(row.get("active", False)),
        "logic_hash": row.get("logic_hash"),
        "regime_profile": row.get("regime_profile"),
        "robustness_score": float(row.get("robustness_score", 0.0) or 0.0),
        "parent_strategy_id": row.get("parent_strategy_id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "validated_at": row.get("validated_at"),
    }


def upsert_strategy(
    strategy_id: str,
    *,
    base_strategy: str = "unknown",
    version: int = 1,
    status: str = "candidate",
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    source: str = "manual",
    notes: str = "",
    active: bool = False,
    validated_at: datetime | str | None = None,
    regime_profile: str | None = None,
    robustness_score: float = 0.0,
    parent_strategy_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create or update a strategy row.

    **kwargs is accepted for backward compatibility with older callers.
    Unknown extras are folded into parameters._meta so callers do not crash.
    """
    store = _load()
    now = _now()

    # Backward-compatible ingestion of extra fields.
    extras = dict(kwargs or {})
    if extras:
        params = dict(parameters or {})
        meta = dict(params.get("_meta") or {})
        meta.update({str(k): _jsonable(v) for k, v in extras.items()})
        params["_meta"] = meta
        parameters = params

    logic_hash = compute_logic_hash(parameters)
    row = {
        "base_strategy": base_strategy,
        "version": int(version or 1),
        "status": status,
        "parameters": _jsonable(parameters or {}),
        "metrics": _jsonable(metrics or {}),
        "tags": _jsonable(tags or []),
        "source": source,
        "notes": notes,
        "active": bool(active),
        "logic_hash": logic_hash,
        "regime_profile": regime_profile,
        "robustness_score": float(robustness_score or 0.0),
        "parent_strategy_id": parent_strategy_id,
        "created_at": store["registry"].get(strategy_id, {}).get("created_at", now),
        "updated_at": now,
        "validated_at": validated_at.isoformat() if hasattr(validated_at, "isoformat") else validated_at,
    }
    store["registry"][strategy_id] = row
    _save(store)
    return _row(strategy_id, row)


def record_experiment(
    strategy_id: str,
    *,
    symbol: str,
    timeframe: str,
    run_type: str = "backtest",
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    passed: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    store = _load()
    store["counters"]["experiment_id"] = int(store["counters"].get("experiment_id", 0)) + 1
    row = {
        "id": store["counters"]["experiment_id"],
        "strategy_id": strategy_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "run_type": run_type,
        "parameters": _jsonable(parameters or {}),
        "metrics": _jsonable(metrics or {}),
        "passed": bool(passed),
        "notes": notes,
        "created_at": _now(),
    }
    store["experiments"].append(row)
    _save(store)
    return row


def record_evolution_run(
    *,
    cycle_id: str,
    symbol: str,
    timeframe: str,
    parent_strategy_id: str | None,
    child_strategy_id: str,
    status: str,
    score: float = 0.0,
    passed: bool = False,
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    store = _load()
    store["counters"]["evolution_id"] = int(store["counters"].get("evolution_id", 0)) + 1
    row = {
        "id": store["counters"]["evolution_id"],
        "cycle_id": cycle_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "parent_strategy_id": parent_strategy_id,
        "child_strategy_id": child_strategy_id,
        "status": status,
        "score": float(score),
        "passed": bool(passed),
        "parameters": _jsonable(parameters or {}),
        "metrics": _jsonable(metrics or {}),
        "notes": notes,
        "created_at": _now(),
    }
    store["evolution_runs"].append(row)
    _save(store)
    return row


def list_strategies(active_only: bool = False) -> list[dict[str, Any]]:
    store = _load()
    rows = [_row(strategy_id, row) for strategy_id, row in store["registry"].items()]
    if active_only:
        rows = [row for row in rows if row.get("active")]
    rows.sort(key=lambda r: (r.get("updated_at") or "", r.get("created_at") or ""), reverse=True)
    return rows


def rank_strategies(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    regime: str | None = None,
    active_only: bool = True,
    limit: int = 10,
) -> List[dict[str, Any]]:
    rows = list_strategies(active_only=active_only)

    def _match(r):
        tags = {str(t).lower() for t in (r.get("tags") or [])}
        if symbol and symbol.lower() not in tags:
            return False
        if timeframe and timeframe.lower() not in tags:
            return False
        if regime and (r.get("regime_profile") or "") != regime:
            return False
        return True

    rows = [r for r in rows if _match(r)]

    def _score(r):
        m = r.get("metrics") or {}
        decision = m.get("decision") or {}
        return (
            float(decision.get("score", 0.0)),
            float(r.get("robustness_score", 0.0)),
            r.get("updated_at") or "",
        )

    rows.sort(key=_score, reverse=True)
    return rows[:limit]


def get_strategy(strategy_id: str) -> dict[str, Any]:
    store = _load()
    return _row(strategy_id, store["registry"].get(strategy_id))


def list_experiments(strategy_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    store = _load()
    rows = [e for e in store["experiments"] if strategy_id is None or e.get("strategy_id") == strategy_id]
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[: max(1, int(limit))]


def list_evolution_runs(strategy_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    store = _load()
    rows = [r for r in store["evolution_runs"] if strategy_id is None or r.get("child_strategy_id") == strategy_id]
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[: max(1, int(limit))]
