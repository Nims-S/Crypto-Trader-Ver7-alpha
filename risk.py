from config import (
    CAPITAL,
    ALLOCATION,
    RISK,
    MAX_SYMBOL_EXPOSURE_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_WEEKLY_LOSS_PCT,
)

# Maximum multiplier applied to initial capital based on historical PnL.
_MAX_CAPITAL_GROWTH = 3.0
_MIN_CAPITAL_GROWTH = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# CAPITAL + PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def get_dynamic_capital(cur, initial_capital: float) -> float:
    cur.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades")
    total_pnl = float(cur.fetchone()[0] or 0.0)
    if initial_capital <= 0:
        return initial_capital
    growth = 1.0 + total_pnl / initial_capital
    growth = max(_MIN_CAPITAL_GROWTH, min(_MAX_CAPITAL_GROWTH, growth))
    return initial_capital * growth


def get_open_exposure(cur) -> float:
    cur.execute("SELECT COALESCE(SUM(entry * size), 0) FROM positions")
    return float(cur.fetchone()[0] or 0.0)


def get_position_count(cur) -> int:
    cur.execute("SELECT COUNT(*) FROM positions")
    return int(cur.fetchone()[0] or 0)


# ─────────────────────────────────────────────────────────────────────────────
# HARD RISK GATES
# ─────────────────────────────────────────────────────────────────────────────

def risk_gate(cur, total_capital: float):
    cur.execute(
        "SELECT COALESCE(SUM(pnl), 0) FROM trades "
        "WHERE timestamp >= NOW() - INTERVAL '1 day'"
    )
    day_pnl = float(cur.fetchone()[0] or 0.0)

    cur.execute(
        "SELECT COALESCE(SUM(pnl), 0) FROM trades "
        "WHERE timestamp >= NOW() - INTERVAL '7 days'"
    )
    week_pnl = float(cur.fetchone()[0] or 0.0)

    if total_capital <= 0:
        return False, "Invalid capital"
    if day_pnl <= -(total_capital * MAX_DAILY_LOSS_PCT):
        return False, "Daily loss limit reached"
    if week_pnl <= -(total_capital * MAX_WEEKLY_LOSS_PCT):
        return False, "Weekly loss limit reached"
    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY KILL SWITCH
# ─────────────────────────────────────────────────────────────────────────────

def get_strategy_pause(cur, strategy: str):
    cur.execute(
        "SELECT paused_until, pause_reason FROM strategy_controls "
        "WHERE strategy=%s AND paused_until > NOW()",
        (strategy,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {"paused_until": row[0], "reason": row[1]}


def evaluate_strategy_pause(
    cur,
    strategy: str,
    min_trades: int = 10,
    min_win_rate: float = 0.4,
    min_profit_factor: float = 0.8,
    pause_hours: int = 24,
    lookback_hours: int = 24,
):
    # already paused?
    active = get_strategy_pause(cur, strategy)
    if active:
        return active

    cur.execute(
        """
        SELECT pnl
        FROM trades
        WHERE strategy=%s
        AND timestamp >= NOW() - (%s * INTERVAL '1 hour')
        ORDER BY timestamp DESC
        """,
        (strategy, lookback_hours),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    pnls = [float(r[0]) for r in rows]
    trades = len(pnls)
    if trades < min_trades:
        return None

    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / trades if trades else 0.0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    if win_rate < min_win_rate or profit_factor < min_profit_factor:
        reason = (
            f"auto_pause: {strategy} trades={trades} "
            f"wr={win_rate:.2f} pf={profit_factor:.2f}"
        )
        cur.execute(
            """
            INSERT INTO strategy_controls(strategy, paused_until, pause_reason, updated_at)
            VALUES (%s, NOW() + (%s * INTERVAL '1 hour'), %s, NOW())
            ON CONFLICT (strategy)
            DO UPDATE SET paused_until=EXCLUDED.paused_until,
                          pause_reason=EXCLUDED.pause_reason,
                          updated_at=NOW()
            """,
            (strategy, pause_hours, reason),
        )
        return {"paused_until": None, "reason": reason}

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL COOLDOWN (ANTI CLUSTER RISK)
# ─────────────────────────────────────────────────────────────────────────────

def get_symbol_cooldown(cur, symbol: str):
    cur.execute(
        "SELECT cooldown_until, cooldown_reason FROM symbol_controls "
        "WHERE symbol=%s AND cooldown_until > NOW()",
        (symbol,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {"cooldown_until": row[0], "reason": row[1]}


def maybe_pause_symbol(cur, symbol: str, loss_streak: int = 2, pause_hours: int = 12):
    cur.execute(
        "SELECT pnl FROM trades WHERE symbol=%s ORDER BY timestamp DESC LIMIT %s",
        (symbol, loss_streak),
    )
    rows = cur.fetchall()
    if len(rows) < loss_streak:
        return None

    pnls = [float(r[0]) for r in rows]
    if not all(p <= 0 for p in pnls):
        return None

    reason = f"cooldown: {symbol} {loss_streak} consecutive losses"
    cur.execute(
        """
        INSERT INTO symbol_controls(symbol, cooldown_until, cooldown_reason, updated_at)
        VALUES (%s, NOW() + (%s * INTERVAL '1 hour'), %s, NOW())
        ON CONFLICT (symbol)
        DO UPDATE SET cooldown_until=EXCLUDED.cooldown_until,
                      cooldown_reason=EXCLUDED.cooldown_reason,
                      updated_at=NOW()
        """,
        (symbol, pause_hours, reason),
    )
    return {"cooldown_until": None, "reason": reason}


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────

def get_strategy_multiplier(cur, strategy: str, regime: str) -> float:
    try:
        cur.execute(
            "SELECT trades, wins, total_pnl FROM strategy_stats "
            "WHERE strategy=%s AND regime=%s",
            (strategy, regime),
        )
        row = cur.fetchone()
        if not row:
            return 1.0
        trades, wins, pnl = row
        if trades < 5:
            return 1.0
        win_rate = wins / trades
        if win_rate > 0.6 and pnl > 0:
            return 1.25
        elif win_rate < 0.4:
            return 0.6
        return 1.0
    except Exception:
        return 1.0


def calculate_position(
    symbol: str,
    price: float,
    total_cap: float,
    stop_loss_pct: float = 0.005,
    confidence: float = 0.5,
    regime_multiplier: float = 1.0,
    size_multiplier: float = 1.0,
):
    ratio = ALLOCATION.get(symbol, 0.33)
    notional_cap = total_cap * ratio

    confidence_multiplier = max(0.5, min(1.25, confidence))
    regime_multiplier     = max(0.5, min(1.5, float(regime_multiplier or 1.0)))
    size_multiplier       = max(0.5, min(1.25, float(size_multiplier or 1.0)))

    risk_budget      = total_cap * RISK * confidence_multiplier * regime_multiplier * size_multiplier
    stop_distance    = max(price * float(stop_loss_pct), price * 0.0025)
    risk_based_size  = risk_budget / stop_distance
    allocation_size  = notional_cap / price

    size              = max(0.0, min(risk_based_size, allocation_size))
    deployed_capital  = size * price

    max_symbol_cap = total_cap * MAX_SYMBOL_EXPOSURE_PCT
    if deployed_capital > max_symbol_cap:
        deployed_capital = max_symbol_cap
        size             = deployed_capital / price

    return size, deployed_capital
