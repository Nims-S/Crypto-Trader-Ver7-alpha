import psycopg2
from config import DB_URL, SYMBOLS


def get_conn():
    return psycopg2.connect(DB_URL, sslmode="require", connect_timeout=10)


def init_db():
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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_controls (
            strategy TEXT PRIMARY KEY,
            paused_until TIMESTAMP,
            pause_reason TEXT DEFAULT '',
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS symbol_controls (
            symbol TEXT PRIMARY KEY,
            cooldown_until TIMESTAMP,
            cooldown_reason TEXT DEFAULT '',
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_stats (
            strategy TEXT,
            regime TEXT,
            trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            total_pnl FLOAT DEFAULT 0,
            last_updated TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (strategy, regime)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_registry (
            strategy_id TEXT PRIMARY KEY,
            base_strategy TEXT NOT NULL DEFAULT 'unknown',
            version INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'candidate',
            parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
            metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
            tags JSONB NOT NULL DEFAULT '[]'::jsonb,
            source TEXT NOT NULL DEFAULT 'manual',
            notes TEXT NOT NULL DEFAULT '',
            active BOOLEAN NOT NULL DEFAULT FALSE,
            validated_at TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_experiments (
            id BIGSERIAL PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            run_type TEXT NOT NULL DEFAULT 'backtest',
            parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
            metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
            passed BOOLEAN NOT NULL DEFAULT FALSE,
            notes TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evolution_runs (
            id BIGSERIAL PRIMARY KEY,
            cycle_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            parent_strategy_id TEXT,
            child_strategy_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'created',
            score FLOAT DEFAULT 0,
            passed BOOLEAN DEFAULT FALSE,
            parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
            metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
            notes TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_experiments_strategy_id ON strategy_experiments(strategy_id, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_experiments_created_at ON strategy_experiments(created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_strategy_registry_active ON strategy_registry(active, updated_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_evolution_runs_cycle_id ON evolution_runs(cycle_id, created_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_evolution_runs_child_id ON evolution_runs(child_strategy_id, created_at DESC)")

    safe_migrations = [
        ("positions", "opened_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("positions", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("positions", "strategy_id_used", "TEXT"),
        ("positions", "strategy_score", "FLOAT DEFAULT 0"),
        ("positions", "was_override_used", "BOOLEAN DEFAULT FALSE"),
        ("trades", "strategy_id_used", "TEXT"),
        ("trades", "strategy_score", "FLOAT DEFAULT 0"),
        ("trades", "was_override_used", "BOOLEAN DEFAULT FALSE"),
        ("strategy_registry", "validated_at", "TIMESTAMP"),
        ("strategy_registry", "created_at", "TIMESTAMP NOT NULL DEFAULT NOW()"),
        ("strategy_registry", "updated_at", "TIMESTAMP NOT NULL DEFAULT NOW()"),
        ("evolution_runs", "created_at", "TIMESTAMP NOT NULL DEFAULT NOW()"),
    ]
    for table, col, col_type in safe_migrations:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type}")

    conn.commit()
    conn.close()
    print("✅ Database schema ready", flush=True)
