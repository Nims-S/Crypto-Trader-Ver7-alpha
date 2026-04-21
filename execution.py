"""
execution.py — position lifecycle management.
"""

from datetime import datetime, timedelta
from utils import send_telegram
from performance import log_trade_performance
from risk import maybe_pause_symbol, evaluate_strategy_pause


def _sl_from_pct(entry: float, pct: float, is_long: bool) -> float:
    return entry * (1 - pct) if is_long else entry * (1 + pct)


def _current_size(cur, symbol: str) -> float:
    cur.execute("SELECT size FROM positions WHERE symbol=%s", (symbol,))
    row = cur.fetchone()
    return float(row[0]) if row else 0.0


def _record_close(cur, symbol, entry, close_price, closed_size, is_long, regime, reason, confidence, strategy):
    if closed_size <= 0:
        return 0.0
    pnl = ((close_price - entry) * closed_size) if is_long else ((entry - close_price) * closed_size)
    cur.execute("INSERT INTO trades (symbol, entry, exit, pnl, regime, reason, confidence, strategy) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (symbol, entry, close_price, pnl, regime, reason, confidence, strategy))
    log_trade_performance(cur, strategy, regime, pnl)
    if pnl <= 0:
        maybe_pause_symbol(cur, symbol)
    return pnl


def open_position(cur, symbol, price, size, deployed_capital, direction, regime, strategy,
                  stop_loss_pct, take_profit_pct, secondary_take_profit_pct,
                  tp3_pct, tp3_close_fraction, trail_pct, trail_atr_mult,
                  tp1_close_fraction, tp2_close_fraction, confidence):
    is_long = direction == "LONG"

    sl = _sl_from_pct(price, stop_loss_pct, is_long)
    sl_dist = abs(price - sl)

    # 1R / 2.5R model
    tp1 = price + sl_dist if is_long else price - sl_dist
    tp2 = price + (sl_dist * 2.5) if is_long else price - (sl_dist * 2.5)

    cur.execute("""
        INSERT INTO positions (
            symbol, entry, sl, tp, tp2, size, original_size,
            regime, confidence, direction, strategy,
            stop_loss_pct, tp1_close_fraction, tp2_close_fraction, opened_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
    """, (
        symbol, price, sl, tp1, tp2, size, size,
        regime, confidence, direction, strategy,
        stop_loss_pct, 0.5, 0.5
    ))

    send_telegram(f"🚀 OPEN {symbol} | Entry={price:.4f} | SL={sl:.4f} | TP1={tp1:.4f} TP2={tp2:.4f}")


def close_position(cur, position: dict, price: float, reason: str):
    symbol = position["symbol"]
    is_long = position["direction"] == "LONG"
    entry = float(position["entry"])
    regime = position.get("regime", "unknown")
    confidence = float(position.get("confidence") or 0)
    strategy = position.get("strategy", "unknown")

    size = _current_size(cur, symbol)
    pnl = _record_close(cur, symbol, entry, price, size, is_long, regime, reason, confidence, strategy)
    cur.execute("DELETE FROM positions WHERE symbol=%s", (symbol,))
    send_telegram(f"🛑 CLOSE {symbol} | Reason={reason} | Exit={price:.4f}")
    return True


def manage_position(cur, position: dict, price: float):
    symbol = position["symbol"]
    is_long = position["direction"] == "LONG"
    entry = float(position["entry"])
    sl = float(position["sl"])
    tp1 = float(position["tp"])
    tp2 = float(position["tp2"])
    tp1_hit = position["tp1_hit"]
    regime = position.get("regime", "unknown")
    confidence = float(position.get("confidence") or 0)
    strategy = position.get("strategy", "unknown")

    # TIME STOP (24h)
    opened_at = position.get("opened_at")
    if opened_at and datetime.utcnow() - opened_at >= timedelta(hours=24):
        close_position(cur, position, price, reason="time_stop")
        return

    # SL
    if (is_long and price <= sl) or (not is_long and price >= sl):
        close_position(cur, position, price, reason="stop_loss")
        return

    # TP1 (partial + BE)
    if not tp1_hit and ((is_long and price >= tp1) or (not is_long and price <= tp1)):
        size = _current_size(cur, symbol)
        close_size = size * 0.5
        _record_close(cur, symbol, entry, price, close_size, is_long, regime, "tp1", confidence, strategy)
        cur.execute("UPDATE positions SET size=%s, sl=%s, tp1_hit=TRUE WHERE symbol=%s",
                    (size - close_size, entry, symbol))
        return

    # TP2 (final)
    if (is_long and price >= tp2) or (not is_long and price <= tp2):
        close_position(cur, position, price, reason="tp2")
