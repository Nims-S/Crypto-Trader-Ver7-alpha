"""Persistence helpers for strategy evolution.

This module prefers PostgreSQL when available, but falls back to a local JSON
store when the database is offline. That keeps the evolution loop usable for
local development and sanity checks without changing the public API.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db import get_conn

_REGISTRY_DEFAULT_STATUS = "candidate"
_REGISTRY_DEFAULT_SOURCE = "manual"
_LOCAL_STORE_PATH = Path(os.getenv("STRATEGY_STORE_FILE", ".strategy_store.json"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(payload: Any, default: Any = None) -> str:
    if payload is None:
        payload = default if default is not None else {}
    return json.dumps(payload, default=str)


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


def _load_store() -> dict[str, Any]:
    if not _LOCAL_STORE_PATH.exists():
        return {"registry": {}, "experiments": [], "evolution_runs": [], "counters": {"experiment_id": 0, "evolution_id": 0}}
    try:
        with _LOCAL_STORE_PATH.open("r", encoding="utf-8") as fh:
            store = json.load(fh)
    except Exception:
        store = {}
    store.setdefault("registry", {})
    store.setdefault("experiments", [])
    store.setdefault("evolution_runs", [])
    store.setdefault("counters", {"experiment_id": 0, "evolution_id": 0})
    return store


def _save_store(store: dict[str, Any]) -> None:
    tmp_path = _LOCAL_STORE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, default=str, sort_keys=True)
    tmp_path.replace(_LOCAL_STORE_PATH)


def _row_to_dict(row) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "strategy_id": row[0],
        "base_strategy": row[1],
        "version": int(row[2] or 1),
        "status": row[3],
        "parameters": row[4] or {},
        "metrics": row[5] or {},
        "tags": row[6] or [],
        "source": row[7],
        "notes": row[8] or "",
        "active": bool(row[9]),
        "created_at": row[10].isoformat() if row[10] else None,
        "updated_at": row[11].isoformat() if row[11] else None,
        "validated_at": row[12].isoformat() if row[12] else None,
    }


def _local_row_from_strategy(strategy_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "base_strategy": row.get("base_strategy", "unknown"),
        "version": int(row.get("version", 1) or 1),
        "status": row.get("status", _REGISTRY_DEFAULT_STATUS),
        "parameters": row.get("parameters", {}) or {},
        "metrics": row.get("metrics", {}) or {},
        "tags": row.get("tags", []) or [],
        "source": row.get("source", _REGISTRY_DEFAULT_SOURCE),
        "notes": row.get("notes", "") or "",
        "active": bool(row.get("active", False)),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "validated_at": row.get("validated_at"),
    }


def _db_available() -> bool:
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


def upsert_strategy(
    strategy_id: str,
    *,
    base_strategy: str = "unknown",
    version: int = 1,
    status: str = _REGISTRY_DEFAULT_STATUS,
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    source: str = _REGISTRY_DEFAULT_SOURCE,
    notes: str = "",
    active: bool = False,
    validated_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Insert or update a strategy candidate."""
    parameters = parameters or {}
    metrics = metrics or {}
    tags = tags or []

    if _db_available():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO strategy_registry (
                strategy_id, base_strategy, version, status,
                parameters, metrics, tags, source, notes,
                active, validated_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s::jsonb, %s, %s,
                %s, %s, NOW()
            )
            ON CONFLICT (strategy_id)
            DO UPDATE SET
                base_strategy = EXCLUDED.base_strategy,
                version = EXCLUDED.version,
                status = EXCLUDED.status,
                parameters = EXCLUDED.parameters,
                metrics = EXCLUDED.metrics,
                tags = EXCLUDED.tags,
                source = EXCLUDED.source,
                notes = EXCLUDED.notes,
                active = EXCLUDED.active,
                validated_at = COALESCE(EXCLUDED.validated_at, strategy_registry.validated_at),
                updated_at = NOW()
            RETURNING
                strategy_id, base_strategy, version, status,
                parameters, metrics, tags, source, notes,
                active, created_at, updated_at, validated_at
            """,
            (
                strategy_id,
                base_strategy,
                int(version or 1),
                status,
                _json_text(parameters, default={}),
                _json_text(metrics, default={}),
                _json_text(tags, default=[]),
                source,
                notes,
                bool(active),
                validated_at,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        conn.close()
        return _row_to_dict(row)

    store = _load_store()
    now = _utc_now_iso()
    row = {
        "base_strategy": base_strategy,
        "version": int(version or 1),
        "status": status,
        "parameters": _jsonable(parameters),
        "metrics": _jsonable(metrics),
        "tags": _jsonable(tags),
        "source": source,
        "notes": notes,
        "active": bool(active),
        "created_at": store["registry"].get(strategy_id, {}).get("created_at", now),
        "updated_at": now,
        "validated_at": validated_at.isoformat() if hasattr(validated_at, "isoformat") else validated_at,
    }
    store["registry"][strategy_id] = row
    _save_store(store)
    return _local_row_from_strategy(strategy_id, row)


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
    """Append a single experiment result to the immutable experiment log."""
    parameters = parameters or {}
    metrics = metrics or {}

    if _db_available():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO strategy_experiments (
                strategy_id, symbol, timeframe, run_type,
                parameters, metrics, passed, notes, created_at
            )
            VALUES (
                %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s, %s, NOW()
            )
            RETURNING
                id, strategy_id, symbol, timeframe, run_type,
                parameters, metrics, passed, notes, created_at
            """,
            (
                strategy_id,
                symbol,
                timeframe,
                run_type,
                _json_text(parameters, default={}),
                _json_text(metrics, default={}),
                bool(passed),
                notes,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        conn.close()

        return {
            "id": int(row[0]),
            "strategy_id": row[1],
            "symbol": row[2],
            "timeframe": row[3],
            "run_type": row[4],
            "parameters": row[5] or {},
            "metrics": row[6] or {},
            "passed": bool(row[7]),
            "notes": row[8] or "",
            "created_at": row[9].isoformat() if row[9] else None,
        }

    store = _load_store()
    store["counters"]["experiment_id"] = int(store["counters"].get("experiment_id", 0)) + 1
    row = {
        "id": store["counters"]["experiment_id"],
        "strategy_id": strategy_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "run_type": run_type,
        "parameters": _jsonable(parameters),
        "metrics": _jsonable(metrics),
        "passed": bool(passed),
        "notes": notes,
        "created_at": _utc_now_iso(),
    }
    store["experiments"].append(row)
    _save_store(store)
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
    """Append a row to the evolution audit log."""
    parameters = parameters or {}
    metrics = metrics or {}

    if _db_available():
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
            RETURNING id, cycle_id, symbol, timeframe, parent_strategy_id, child_strategy_id,
                      status, score, passed, parameters, metrics, notes, created_at
            """,
            (
                cycle_id,
                symbol,
                timeframe,
                parent_strategy_id,
                child_strategy_id,
                status,
                score,
                passed,
                _json_text(parameters, default={}),
                _json_text(metrics, default={}),
                notes,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        conn.close()
        return {
            "id": int(row[0]),
            "cycle_id": row[1],
            "symbol": row[2],
            "timeframe": row[3],
            "parent_strategy_id": row[4],
            "child_strategy_id": row[5],
            "status": row[6],
            "score": float(row[7] or 0.0),
            "passed": bool(row[8]),
            "parameters": row[9] or {},
            "metrics": row[10] or {},
            "notes": row[11] or "",
            "created_at": row[12].isoformat() if row[12] else None,
        }

    store = _load_store()
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
        "parameters": _jsonable(parameters),
        "metrics": _jsonable(metrics),
        "notes": notes,
        "created_at": _utc_now_iso(),
    }
    store["evolution_runs"].append(row)
    _save_store(store)
    return row


def list_strategies(active_only: bool = False) -> list[dict[str, Any]]:
    if _db_available():
        conn = get_conn()
        cur = conn.cursor()
        query = """
            SELECT strategy_id, base_strategy, version, status,
                   parameters, metrics, tags, source, notes,
                   active, created_at, updated_at, validated_at
            FROM strategy_registry
        """
        if active_only:
            query += " WHERE active = TRUE"
        query += " ORDER BY updated_at DESC NULLS LAST, created_at DESC"

        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        return [_row_to_dict(row) for row in rows]

    store = _load_store()
    rows = [
        _local_row_from_strategy(strategy_id, row)
        for strategy_id, row in store["registry"].items()
        if not active_only or bool(row.get("active", False))
    ]
    rows.sort(key=lambda r: (r.get("updated_at") or "", r.get("created_at") or ""), reverse=True)
    return rows


def get_strategy(strategy_id: str) -> dict[str, Any]:
    if _db_available():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT strategy_id, base_strategy, version, status,
                   parameters, metrics, tags, source, notes,
                   active, created_at, updated_at, validated_at
            FROM strategy_registry
            WHERE strategy_id = %s
            """,
            (strategy_id,),
        )
        row = cur.fetchone()
        conn.close()
        return _row_to_dict(row)

    store = _load_store()
    row = store["registry"].get(strategy_id)
    return _local_row_from_strategy(strategy_id, row) if row else {}


def list_experiments(strategy_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    if _db_available():
        conn = get_conn()
        cur = conn.cursor()

        if strategy_id:
            cur.execute(
                """
                SELECT id, strategy_id, symbol, timeframe, run_type,
                       parameters, metrics, passed, notes, created_at
                FROM strategy_experiments
                WHERE strategy_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (strategy_id, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, strategy_id, symbol, timeframe, run_type,
                       parameters, metrics, passed, notes, created_at
                FROM strategy_experiments
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )

        rows = cur.fetchall()
        conn.close()
        return [
            {
                "id": int(row[0]),
                "strategy_id": row[1],
                "symbol": row[2],
                "timeframe": row[3],
                "run_type": row[4],
                "parameters": row[5] or {},
                "metrics": row[6] or {},
                "passed": bool(row[7]),
                "notes": row[8] or "",
                "created_at": row[9].isoformat() if row[9] else None,
            }
            for row in rows
        ]

    store = _load_store()
    rows = [e for e in store["experiments"] if strategy_id is None or e.get("strategy_id") == strategy_id]
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[: max(1, int(limit))]


def list_evolution_runs(strategy_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    if _db_available():
        conn = get_conn()
        cur = conn.cursor()
        if strategy_id:
            cur.execute(
                """
                SELECT id, cycle_id, symbol, timeframe, parent_strategy_id, child_strategy_id,
                       status, score, passed, parameters, metrics, notes, created_at
                FROM evolution_runs
                WHERE child_strategy_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (strategy_id, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, cycle_id, symbol, timeframe, parent_strategy_id, child_strategy_id,
                       status, score, passed, parameters, metrics, notes, created_at
                FROM evolution_runs
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "id": int(row[0]),
                "cycle_id": row[1],
                "symbol": row[2],
                "timeframe": row[3],
                "parent_strategy_id": row[4],
                "child_strategy_id": row[5],
                "status": row[6],
                "score": float(row[7] or 0.0),
                "passed": bool(row[8]),
                "parameters": row[9] or {},
                "metrics": row[10] or {},
                "notes": row[11] or "",
                "created_at": row[12].isoformat() if row[12] else None,
            }
            for row in rows
        ]

    store = _load_store()
    rows = [r for r in store["evolution_runs"] if strategy_id is None or r.get("child_strategy_id") == strategy_id]
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[: max(1, int(limit))]
