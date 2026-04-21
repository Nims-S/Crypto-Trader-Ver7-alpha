"""
Trade performance tracker.

IMPORTANT: log_trade_performance() takes the caller's cursor and does NOT
commit. Commit is handled by the caller's transaction so that strategy_stats
updates are always rolled back together with the trade INSERT on error.
"""


def log_trade_performance(cur, strategy, regime, pnl):
    is_win = 1 if pnl > 0 else 0
    cur.execute("""
        INSERT INTO strategy_stats (strategy, regime, trades, wins, total_pnl)
        VALUES (%s, %s, 1, %s, %s)
        ON CONFLICT (strategy, regime)
        DO UPDATE SET
            trades       = strategy_stats.trades + 1,
            wins         = strategy_stats.wins + %s,
            total_pnl    = strategy_stats.total_pnl + %s,
            last_updated = NOW()
    """, (strategy, regime, is_win, pnl, is_win, pnl))
