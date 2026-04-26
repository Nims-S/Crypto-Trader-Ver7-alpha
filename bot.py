from __future__ import annotations

import os, time
from copy import deepcopy
from datetime import datetime

import ccxt
import pandas as pd

from backtest import run_backtest as run_shadow_backtest
from caffeine import push_to_caffeine
from config import CAPITAL, CANDLE_LIMIT, ENABLE_REGISTRY_STRATEGIES, MAX_COOLDOWN_SECONDS, MAX_POSITIONS, SYMBOLS
from db import get_conn
from execution import manage_position, open_position
from price_feed import feeds
from risk import calculate_position, get_dynamic_capital, get_strategy_multiplier, get_symbol_cooldown, risk_gate
from state import get_controls, get_state, update_asset
from strategy import StrategyState, compute_indicators, generate_signal
from strategy_selector import build_effective_state, select_strategy, strategy_runtime_summary

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
try:
    exchange.load_markets()
except Exception as e:
    print(f"[EXCHANGE WARN] load_markets failed: {e}", flush=True)

ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "false").strip().lower() in {"1", "true", "yes", "on"}
ROUTE_RECHECK_SECONDS = int(os.getenv("ROUTE_RECHECK_SECONDS", 6 * 60 * 60))
ROUTE_LOOKBACK_BARS = int(os.getenv("ROUTE_LOOKBACK_BARS", 500))
ROUTE_ENABLE_MIN_TRADES = int(os.getenv("ROUTE_ENABLE_MIN_TRADES", 20))
ROUTE_ENABLE_MIN_PF = float(os.getenv("ROUTE_ENABLE_MIN_PF", 1.15))
ROUTE_ENABLE_MIN_WR = float(os.getenv("ROUTE_ENABLE_MIN_WR", 0.55))
ROUTE_DISABLE_MAX_PF = float(os.getenv("ROUTE_DISABLE_MAX_PF", 0.95))
ROUTE_DISABLE_MIN_WR = float(os.getenv("ROUTE_DISABLE_MIN_WR", 0.48))
ROUTE_DISABLE_MAX_DD = float(os.getenv("ROUTE_DISABLE_MAX_DD", -8.0))

DEFAULT_ROUTE = {"enabled": True, "ltf_timeframe": "4h", "htf_timeframe": "1d", "allow_shorts": False, "strategy_mode": "alt"}
SYMBOL_ROUTES = {
    "BTC/USDT": {"enabled": True, "ltf_timeframe": "4h", "htf_timeframe": "1d", "allow_shorts": ALLOW_SHORTS, "strategy_mode": "vetf"},
    "SOL/USDT": {"enabled": True, "ltf_timeframe": "4h", "htf_timeframe": "1d", "allow_shorts": False, "strategy_mode": "alt"},
    "ETH/USDT": {"enabled": False, "ltf_timeframe": "4h", "htf_timeframe": "1d", "allow_shorts": False, "strategy_mode": "alt", "pause_reason": "awaiting_edge"},
}

_candle_cache, CANDLE_CACHE_TTL, _route_runtime, _route_last_eval = {}, 60, {}, {}


def _route_for_symbol(symbol):
    route = deepcopy(DEFAULT_ROUTE)
    route.update(SYMBOL_ROUTES.get(symbol, {}))
    return route


def _should_recheck_route(symbol): return (time.time() - _route_last_eval.get(symbol, 0.0)) >= ROUTE_RECHECK_SECONDS


def _evaluate_symbol_edge(symbol, route):
    try:
        result = run_shadow_backtest(symbol=symbol, tf=route["ltf_timeframe"], start=None, end=None, allow_shorts=bool(route.get("allow_shorts", False)), max_bars=ROUTE_LOOKBACK_BARS, use_cache=True)
        return result if isinstance(result, dict) and not result.get("error") else None
    except Exception as exc:
        print(f"[ADAPTIVE ROUTE ERROR] {symbol}: {exc}", flush=True)
        return None


def _adaptive_route_update(symbol, route):
    if symbol != "ETH/USDT" or not _should_recheck_route(symbol): return route
    _route_last_eval[symbol] = time.time()
    score = _evaluate_symbol_edge(symbol, route)
    if not score: return route
    trades, pf, wr, dd = int(score.get("trades", 0) or 0), float(score.get("profit_factor", 0.0) or 0.0), float(score.get("win_rate", 0.0) or 0.0), float(score.get("max_drawdown_pct", 0.0) or 0.0)
    should_enable = trades >= ROUTE_ENABLE_MIN_TRADES and pf >= ROUTE_ENABLE_MIN_PF and wr >= ROUTE_ENABLE_MIN_WR and dd > ROUTE_DISABLE_MAX_DD
    should_disable = trades >= ROUTE_ENABLE_MIN_TRADES and (pf < ROUTE_DISABLE_MAX_PF or wr < ROUTE_DISABLE_MIN_WR or dd <= ROUTE_DISABLE_MAX_DD)
    if not route.get("enabled", False) and should_enable:
        route["enabled"], route["pause_reason"] = True, "edge_restored"
    elif route.get("enabled", False) and should_disable:
        route["enabled"], route["pause_reason"] = False, f"edge_faded pf={pf:.2f} wr={wr:.2f} dd={dd:.2f}"
    print(f"[ADAPTIVE ROUTE] {symbol} | enabled={route.get('enabled', False)} | trades={trades} pf={pf:.2f} wr={wr:.2f} dd={dd:.2f}", flush=True)
    return route


def fetch_historical_data(symbol, timeframe):
    key = f"{symbol}_{timeframe}"
    ts, df = _candle_cache.get(key, (0.0, pd.DataFrame()))
    if not df.empty and (time.time() - ts) < CANDLE_CACHE_TTL: return df
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
        if not bars: return pd.DataFrame()
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = compute_indicators(df)
        _candle_cache[key] = (time.time(), df)
        return df
    except Exception as e:
        print(f"[FETCH ERROR] {symbol} {timeframe}: {e}", flush=True)
        return df


def load_position(cur, symbol):
    cur.execute("""
        SELECT symbol, entry, sl, tp, tp2, tp3, size, original_size, regime, confidence, direction,
               tp1_hit, tp2_hit, tp3_hit, strategy, stop_loss_pct, take_profit_pct, secondary_take_profit_pct,
               trail_pct, trail_atr_mult, tp1_close_fraction, tp2_close_fraction, tp3_close_fraction,
               strategy_id_used, strategy_score, was_override_used, opened_at
        FROM positions WHERE symbol=%s FOR UPDATE SKIP LOCKED
    """, (symbol,))
    row = cur.fetchone()
    if not row: return None
    return {"symbol": row[0], "entry": row[1], "sl": row[2], "tp": row[3], "tp2": row[4], "tp3": row[5], "size": float(row[6]), "original_size": float(row[7] or row[6]), "regime": row[8], "confidence": row[9], "direction": row[10], "tp1_hit": row[11], "tp2_hit": row[12], "tp3_hit": row[13], "strategy": row[14], "stop_loss_pct": row[15], "take_profit_pct": row[16], "secondary_take_profit_pct": row[17], "trail_pct": row[18], "trail_atr_mult": row[19], "tp1_close_fraction": row[20], "tp2_close_fraction": row[21], "tp3_close_fraction": row[22], "strategy_id_used": row[23], "strategy_score": row[24], "was_override_used": row[25], "opened_at": row[26]}


def build_position_state(position):
    if not position: return None
    opened_at = position.get("opened_at")
    if hasattr(opened_at, "isoformat"): opened_at = opened_at.isoformat()
    return {"entry_price": position["entry"], "stop_loss": position["sl"], "take_profit": position["tp"], "take_profit_2": position["tp2"], "take_profit_3": position.get("tp3"), "size": position["size"], "original_size": position.get("original_size"), "strategy": position["strategy"], "strategy_id_used": position.get("strategy_id_used"), "strategy_score": position.get("strategy_score"), "was_override_used": position.get("was_override_used"), "opened_at": opened_at}


def _latest_closed_slice(df): return pd.DataFrame() if df is None or df.empty or len(df) < 3 else df.iloc[:-1].copy().reset_index(drop=True)

def _to_float(value):
    try: return float(value)
    except Exception: return None

def _log_route(symbol, route): print(f"[ROUTE] {symbol} | enabled={route['enabled']} | ltf={route['ltf_timeframe']} | htf={route['htf_timeframe']} | mode={route['strategy_mode']}", flush=True)


def _strategy_context(symbol, route, base_state):
    fallback = f"legacy_{symbol.replace('/', '_').lower()}"
    if not ENABLE_REGISTRY_STRATEGIES: return fallback, None, base_state
    strategy_id, strategy_meta = select_strategy(symbol=symbol, timeframe=route["ltf_timeframe"], fallback=fallback)
    return strategy_id, strategy_meta, build_effective_state(base_state, strategy_meta)


def run_bot():
    print("[BOT] LOOP STARTED (routed closed-candle mode)", flush=True)
    last_trade_time, last_signal_candle = {}, {}
    routes = {s: _route_runtime.setdefault(s, _route_for_symbol(s)) for s in SYMBOLS}
    states = {s: StrategyState(allow_shorts=bool(routes[s].get("allow_shorts", False))) for s in SYMBOLS}
    while True:
        print(f"[HEARTBEAT] Bot alive at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
        conn = cur = None
        try:
            conn = get_conn(); cur = conn.cursor()
            total_cap = get_dynamic_capital(cur, CAPITAL)
            allowed, reason = risk_gate(cur, total_cap)
            if not allowed:
                print(f"[RISK BLOCK] {reason}", flush=True); conn.commit(); time.sleep(3); continue
            controls = get_controls(); global_enabled = controls.get("GLOBAL", {}).get("enabled", True)
            for symbol in SYMBOLS:
                route = _adaptive_route_update(symbol, routes[symbol]); _log_route(symbol, route)
                feed = feeds.get(symbol)
                if feed is None: continue
                try: price = feed.get_price()
                except Exception as e: print(f"[FEED ERROR] {symbol}: {e}", flush=True); continue
                if price is None or price <= 0: continue
                position = load_position(cur, symbol)
                if position:
                    try:
                        df_manage = fetch_historical_data(symbol, route["ltf_timeframe"])
                        current_atr = current_ema20 = None
                        if not df_manage.empty:
                            closed_manage = _latest_closed_slice(df_manage)
                            if not closed_manage.empty:
                                current_atr = _to_float(closed_manage.iloc[-1].get("atr")); current_ema20 = _to_float(closed_manage.iloc[-1].get("ema20"))
                        manage_position(cur, position, price, current_atr, current_ema20)
                        position = load_position(cur, symbol)
                    except Exception as e:
                        print(f"[MANAGE ERROR] {symbol}: {e}", flush=True)
                blocked = (not global_enabled) or (not controls.get(symbol, {}).get("enabled", True)) or (not route.get("enabled", True))
                if blocked:
                    update_asset(symbol=symbol, regime="paused", strategy=route.get("pause_reason", "route_disabled") if not route.get("enabled", True) else "kill_switch", signal=None, position=build_position_state(position)); continue
                if get_symbol_cooldown(cur, symbol):
                    update_asset(symbol=symbol, regime="paused", strategy="symbol_cooldown", signal=None, position=build_position_state(position)); continue
                df = fetch_historical_data(symbol, route["ltf_timeframe"])
                if df.empty:
                    update_asset(symbol=symbol, regime="unknown", strategy="data_unavailable", signal=None, position=build_position_state(position)); continue
                closed_df = _latest_closed_slice(df)
                if closed_df.empty:
                    update_asset(symbol=symbol, regime="unknown", strategy="waiting_for_close", signal=None, position=build_position_state(position)); continue
                df_htf = fetch_historical_data(symbol, route["htf_timeframe"])
                candle_ts = closed_df.iloc[-1]["timestamp"]
                if last_signal_candle.get(symbol) == candle_ts:
                    update_asset(symbol=symbol, regime=position["regime"] if position else "watching", strategy=position["strategy"] if position else "waiting_for_new_candle", signal=None, position=build_position_state(position)); continue
                strategy_id_used, strategy_meta, effective_state = _strategy_context(symbol, route, states[symbol])
                signal = generate_signal(closed_df, state=effective_state, symbol=symbol, df_htf=df_htf, strategy_override=strategy_meta)
                last_signal_candle[symbol] = candle_ts
                if signal and signal.strategy != "no_trade": print(f"[SIGNAL] {symbol} | {signal.strategy} | regime={signal.regime} | conf={signal.confidence:.2f}", flush=True)
                runtime_meta = strategy_runtime_summary(strategy_meta)
                update_asset(symbol=symbol, regime=signal.regime if signal else "unknown", strategy=signal.strategy if signal else "none", signal={"side": signal.side if signal else None, "confidence": getattr(signal, "confidence", None), "strategy_id_used": strategy_id_used, "strategy_score": runtime_meta.get("strategy_score", 0.0), "was_override_used": runtime_meta.get("was_override_used", False)} if signal else None, position=build_position_state(position))
                if signal and signal.strategy != "no_trade" and not position:
                    cur.execute("SELECT COUNT(*) FROM positions"); active_trades = int(cur.fetchone()[0] or 0)
                    if active_trades >= MAX_POSITIONS: continue
                    now = time.time()
                    if symbol in last_trade_time and (now - last_trade_time[symbol] < MAX_COOLDOWN_SECONDS): continue
                    strategy_mult = get_strategy_multiplier(cur, signal.strategy, signal.regime)
                    size, deployed = calculate_position(symbol=symbol, price=price, total_cap=total_cap, stop_loss_pct=signal.stop_loss_pct, confidence=signal.confidence, regime_multiplier=strategy_mult, size_multiplier=float(getattr(signal, "size_multiplier", 1.0) or 1.0))
                    if size and size > 0:
                        open_position(cur=cur, symbol=symbol, price=price, size=size, deployed_capital=deployed, direction=signal.side, regime=signal.regime, strategy=signal.strategy, stop_loss_pct=signal.stop_loss_pct, take_profit_pct=signal.take_profit_pct, secondary_take_profit_pct=signal.secondary_take_profit_pct, tp3_pct=signal.tp3_pct, tp3_close_fraction=signal.tp3_close_fraction, trail_pct=signal.trail_pct, trail_atr_mult=signal.trail_atr_mult, tp1_close_fraction=signal.tp1_close_fraction, tp2_close_fraction=signal.tp2_close_fraction, confidence=signal.confidence, strategy_id_used=strategy_id_used, strategy_score=runtime_meta.get("strategy_score", 0.0), was_override_used=runtime_meta.get("was_override_used", False))
                        print(f"[ENTRY] {symbol}", flush=True); last_trade_time[symbol] = now
            conn.commit()
            try:
                state = get_state()
                if state.get("assets"): push_to_caffeine(state)
            except Exception as e:
                print(f"[CAFFEINE ERROR] {e}", flush=True)
        except Exception as e:
            if conn: conn.rollback()
            print(f"[CRITICAL ERROR] {e}", flush=True)
        finally:
            if cur: cur.close()
            if conn: conn.close()
        time.sleep(3)


if __name__ == "__main__": run_bot()
