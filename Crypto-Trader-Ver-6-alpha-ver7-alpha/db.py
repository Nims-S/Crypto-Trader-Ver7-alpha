import psycopg2
from config import DB_URL, SYMBOLS


def get_conn():
    return psycopg2.connect(DB_URL, sslmode="require", connect_timeout=10)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol                      TEXT PRIMARY KEY,
            entry                       FLOAT,
            sl                          FLOAT,
            tp                          FLOAT,
            tp2                         FLOAT,
            tp3                         NUMERIC,
            size                        FLOAT,
            original_size               NUMERIC,
            regime                      TEXT    DEFAULT 'unknown',
            confidence                  FLOAT   DEFAULT 0,
            direction                   TEXT    DEFAULT 'LONG',
            strategy                    TEXT    DEFAULT 'unknown',
            stop_loss_pct               FLOAT   DEFAULT 0,
            take_profit_pct             FLOAT   DEFAULT 0,
            secondary_take_profit_pct   FLOAT   DEFAULT 0,
            trail_pct                   FLOAT   DEFAULT 0,
            trail_atr_mult              FLOAT   DEFAULT 0,
            tp1_close_fraction          FLOAT   DEFAULT 0.33,
            tp2_close_fraction          FLOAT   DEFAULT 0.5,
            tp3_close_fraction          NUMERIC,
            tp1_hit                     BOOLEAN DEFAULT FALSE,
            tp2_hit                     BOOLEAN DEFAULT FALSE,
            tp3_hit                     BOOLEAN DEFAULT FALSE,
            updated_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT size_non_negative CHECK (size >= 0)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          BIGSERIAL PRIMARY KEY,
            symbol      TEXT      NOT NULL,
            entry       FLOAT     NOT NULL,
            exit        FLOAT     NOT NULL,
            pnl         FLOAT     NOT NULL DEFAULT 0,
            regime      TEXT      DEFAULT 'unknown',
            reason      TEXT      DEFAULT '',
            confidence  FLOAT     DEFAULT 0,
            strategy    TEXT      DEFAULT 'unknown',
            timestamp   TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp
        ON trades(timestamp DESC)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS asset_state (
            symbol     TEXT PRIMARY KEY,
            regime     TEXT,
            strategy   TEXT,
            signal     JSONB,
            position   JSONB,
            updated_at TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_controls (
            scope              TEXT PRIMARY KEY,
            enabled            BOOLEAN NOT NULL DEFAULT TRUE,
            flatten_on_disable BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at         TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    for scope in ["GLOBAL"] + list(SYMBOLS):
        cur.execute("""
            INSERT INTO trade_controls (scope, enabled, flatten_on_disable)
            VALUES (%s, TRUE, FALSE)
            ON CONFLICT (scope) DO NOTHING
        """, (scope,))

    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_controls (
            strategy      TEXT PRIMARY KEY,
            paused_until  TIMESTAMP,
            pause_reason  TEXT DEFAULT '',
            updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS symbol_controls (
            symbol        TEXT PRIMARY KEY,
            cooldown_until TIMESTAMP,
            cooldown_reason TEXT DEFAULT '',
            updated_at    TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_stats (
            strategy     TEXT,
            regime       TEXT,
            trades       INTEGER   DEFAULT 0,
            wins         INTEGER   DEFAULT 0,
            total_pnl    FLOAT     DEFAULT 0,
            last_updated TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (strategy, regime)
        )
    """)

    safe_migrations = [
        ("positions", "tp2",                       "FLOAT"),
        ("positions", "tp3",                       "NUMERIC"),
        ("positions", "direction",                 "TEXT DEFAULT 'LONG'"),
        ("positions", "strategy",                  "TEXT DEFAULT 'unknown'"),
        ("positions", "stop_loss_pct",             "FLOAT DEFAULT 0"),
        ("positions", "take_profit_pct",           "FLOAT DEFAULT 0"),
        ("positions", "secondary_take_profit_pct", "FLOAT DEFAULT 0"),
        ("positions", "trail_pct",                 "FLOAT DEFAULT 0"),
        ("positions", "trail_atr_mult",            "FLOAT DEFAULT 0"),
        ("positions", "tp1_close_fraction",        "FLOAT DEFAULT 0.33"),
        ("positions", "tp2_close_fraction",        "FLOAT DEFAULT 0.5"),
        ("positions", "tp3_close_fraction",        "NUMERIC"),
        ("positions", "original_size",             "NUMERIC"),
        ("positions", "tp1_hit",                   "BOOLEAN DEFAULT FALSE"),
        ("positions", "tp2_hit",                   "BOOLEAN DEFAULT FALSE"),
        ("positions", "tp3_hit",                   "BOOLEAN DEFAULT FALSE"),
        ("positions", "regime",                    "TEXT DEFAULT 'unknown'"),
        ("positions", "confidence",                "FLOAT DEFAULT 0"),
        ("positions", "updated_at",                "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("trades",    "regime",                    "TEXT DEFAULT 'unknown'"),
        ("trades",    "reason",                    "TEXT DEFAULT ''"),
        ("trades",    "confidence",                "FLOAT DEFAULT 0"),
        ("trades",    "strategy",                  "TEXT DEFAULT 'unknown'"),
    ]
    for table, col, col_type in safe_migrations:
        cur.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type}"
        )

    conn.commit()
    conn.close()
    print("✅ Database schema ready", flush=True)
