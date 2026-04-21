# state.py (NEW FILE)
from datetime import datetime

import json

from config import SYMBOLS
from db import get_conn

print("✅ STATE.PY LOADED", flush=True)
STATE = {
    "last_update": None,
    "assets": {}
}

CONTROL_SCOPES = {"GLOBAL", *SYMBOLS}


def normalize_scope(scope):
    return (scope or "GLOBAL").strip().upper()


def _default_control_row():
    return {
        "enabled": True,
        "flatten_on_disable": False,
        "updated_at": None,
    }


def _validate_scope(scope):
    scope = normalize_scope(scope)
    if scope not in CONTROL_SCOPES:
        raise ValueError(f"unknown control scope: {scope}")
    return scope


def update_asset(symbol, regime, strategy, signal=None, position=None):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO asset_state (symbol, regime, strategy, signal, position, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (symbol) DO UPDATE SET
            regime = EXCLUDED.regime,
            strategy = EXCLUDED.strategy,
            signal = EXCLUDED.signal,
            position = EXCLUDED.position,
            updated_at = NOW()
    """, (
        symbol,
        regime,
        strategy,
        json.dumps(signal, default=str) if signal else None,
        json.dumps(position, default=str) if position else None
    ))
    print(f"[STATE UPDATE] {symbol} | regime={regime} | strategy={strategy}", flush=True)

    conn.commit()
    conn.close()


def get_controls():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT scope, enabled, flatten_on_disable, updated_at
        FROM trade_controls
        ORDER BY scope
    """)
    rows = cur.fetchall()
    conn.close()

    controls = {
        r[0]: {
            "enabled": r[1],
            "flatten_on_disable": r[2],
            "updated_at": r[3].isoformat() if r[3] else None,
        }
        for r in rows
    }

    for scope in CONTROL_SCOPES:
        controls.setdefault(scope, _default_control_row())

    return controls


def set_control(scope, enabled=None, flatten_on_disable=None):
    scope = _validate_scope(scope)

    if enabled is None and flatten_on_disable is None:
        return get_controls()

    # Use concrete INSERT values so a partial update on a brand-new scope still
    # satisfies NOT NULL columns while the UPDATE branch preserves existing data.
    insert_enabled = True if enabled is None else bool(enabled)
    insert_flatten = False if flatten_on_disable is None else bool(flatten_on_disable)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO trade_controls (scope, enabled, flatten_on_disable)
        VALUES (%s, %s, %s)
        ON CONFLICT (scope)
        DO UPDATE SET
            enabled = COALESCE(%s, trade_controls.enabled),
            flatten_on_disable = COALESCE(%s, trade_controls.flatten_on_disable),
            updated_at = NOW()
    """, (scope, insert_enabled, insert_flatten, enabled, flatten_on_disable))

    conn.commit()
    conn.close()
    return get_controls()


def get_state():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT symbol, regime, strategy, signal, position, updated_at FROM asset_state")

    rows = cur.fetchall()

    assets = {}
    latest_update = None
    for r in rows:
        assets[r[0]] = {
            "regime": r[1],
            "strategy": r[2],
            "signal": r[3],
            "position": r[4],
            "timestamp": r[5].isoformat() if r[5] else None
        }
        if r[5] and (latest_update is None or r[5] > latest_update):
            latest_update = r[5]
    conn.close()
    return {
        "assets": assets,
        "controls": get_controls(),
        "last_update": latest_update.isoformat() if latest_update else None
    }
