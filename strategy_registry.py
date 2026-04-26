"""Persistence helpers for strategy evolution.

This module stores a canonical registry of candidate strategies and a separate
experiment log for backtests / paper-trading runs. It is intentionally kept
small so future iterations can plug in richer scoring, promotion, and rollback
logic without changing live execution code paths.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from db import get_conn

_REGISTRY_DEFAULT_STATUS = "candidate"
_REGISTRY_DEFAULT_SOURCE = "manual"


def _json_text(payload: Any, default: Any = None) -> str:
    if payload is None:
        payload = default if default is not None else {}
    return json.dumps(payload, default=str)


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


def list_strategies(active_only: bool = False) -> list[dict[str, Any]]:
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


def get_strategy(strategy_id: str) -> dict[str, Any]:
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


def list_experiments(strategy_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
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
