from __future__ import annotations

import os
from typing import Any

from config import DB_URL, SYMBOLS

try:
    import psycopg2  # type: ignore
except Exception:
    psycopg2 = None  # type: ignore


def _local_only() -> bool:
    return os.getenv("LOCAL_STORE_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}


def get_conn():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Set LOCAL_STORE_ONLY=1 for offline evolution runs.")
    if not DB_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg2.connect(DB_URL, sslmode="require", connect_timeout=10)


def init_db():
    if psycopg2 is None or _local_only() or not DB_URL:
        print("[WARN] DB init skipped (local mode or psycopg2 unavailable)", flush=True)
        return

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol TEXT PRIMARY KEY,
            entry FLOAT,
            sl FLOAT,
            tp FLOAT,
            tp2 FLOAT,
            tp3 NUMERIC,
            size FLOAT,
            original_size NUMERIC,
            regime TEXT DEFAULT 'unknown',
            confidence FLOAT DEFAULT 0,
            direction TEXT DEFAULT 'LONG',
            strategy TEXT DEFAULT 'unknown',
            strategy_id_used TEXT,
            strategy_score FLOAT DEFAULT 0,
            was_override_used BOOLEAN DEFAULT FALSE,
            stop_loss_pct FLOAT DEFAULT 0,
            take_profit_pct FLOAT DEFAULT 0,
            secondary_take_profit_pct FLOAT DEFAULT 0,
            trail_pct FLOAT DEFAULT 0,
            trail_atr_mult FLOAT DEFAULT 0,
            tp1_close_fraction FLOAT DEFAULT 0.33,
            tp2_close_fraction FLOAT DEFAULT 0.5,
            tp3_close_fraction NUMERIC,
            tp1_hit BOOLEAN DEFAULT FALSE,
            tp2_hit BOOLEAN DEFAULT FALSE,
            tp3_hit BOOLEAN DEFAULT FALSE,
            opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT size_non_negative CHECK (size >= 0)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            entry FLOAT NOT NULL,
            exit FLOAT NOT NULL,
            pnl FLOAT NOT NULL DEFAULT 0,
            regime TEXT DEFAULT 'unknown',
            reason TEXT DEFAULT '',
            confidence FLOAT DEFAULT 0,
            strategy TEXT DEFAULT 'unknown',
            strategy_id_used TEXT,
            strategy_score FLOAT DEFAULT 0,
            was_override_used BOOLEAN DEFAULT FALSE,
            timestamp TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS asset_state (
            symbol TEXT PRIMARY KEY,
            regime TEXT,
            strategy TEXT,
            signal JSONB,
            position JSONB,
            updated_at TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_controls (
            scope TEXT PRIMARY KEY,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            flatten_on_disable BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    for scope in ["GLOBAL"] + list(SYMBOLS):
        cur.execute("INSERT INTO trade_controls (scope, enabled, flatten_on_disable) VALUES (%s, TRUE, FALSE) ON CONFLICT (scope) DO NOTHING", (scope,))

    conn.commit()
    conn.close()
    print("✅ Database schema ready", flush=True)
