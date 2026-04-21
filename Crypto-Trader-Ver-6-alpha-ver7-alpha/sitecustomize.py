"""Runtime patch layer for deterministic control semantics.

Python imports sitecustomize automatically when it is present on sys.path.
This lets us override the bot loop without disturbing the rest of the module
layout or the Render entrypoint.
"""

from __future__ import annotations

import time
from datetime import datetime

try:
    import bot as _bot
    from execution import close_position, manage_position, open_position, update_position_levels
    from risk import calculate_position, get_dynamic_capital, get_strategy_multiplier, risk_gate
    from state import get_controls, get_state, update_asset
except Exception as _import_error:  # pragma: no cover - boot-time safety guard
    print(f"[SITECUSTOMIZE] import skipped: {_import_error}", flush=True)
else:
    def _build_position_state(position):
        if not position:
            return None
        return {
            "entry_price": position["entry"],
            "stop_loss": position["sl"],
            "take_profit": position["tp"],
            "take_profit_2": position["tp2"],
            "take_profit_3": position.get("tp3"),
            "size": position["size"],
            "original_size": position.get("original_size"),
            "strategy": position["strategy"],
        }

    def _patched_run_bot():
        print("[BOT] LOOP STARTED (v6 beta control patch)", flush=True)
        last_trade_time: dict[str, float] = {}

        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[HEARTBEAT] Bot alive at {timestamp}", flush=True)

            conn = None
            cur = None
            try:
                conn = _bot.get_conn()
                cur = conn.cursor()

                total_cap = get_dynamic_capital(cur, _bot.CAPITAL)
                allowed, reason = risk_gate(cur, total_cap)
                if not allowed:
                    print(f"[RISK BLOCK] {reason}", flush=True)
                    conn.commit()
                    time.sleep(3)
                    continue

                for symbol in _bot.SYMBOLS:
                    feed = _bot.feeds.get(symbol)
                    if feed is None:
                        continue

                    try:
                        price = feed.get_price()
                    except Exception as exc:
                        print(f"[FEED ERROR] {symbol}: {exc}", flush=True)
                        continue

                    if price is None or price <= 0:
                        continue

                    position = _bot.load_position(cur, symbol)
                    if position:
                        try:
                            update_position_levels(
                                cur,
                                symbol,
                                position.get("stop_loss_pct", 0),
                                position.get("take_profit_pct", 0),
                                position.get("secondary_take_profit_pct", 0),
                                None,
                            )
                        except Exception as exc:
                            print(f"[SYNC ERROR] {symbol}: {exc}", flush=True)

                    controls = get_controls()
                    global_ctrl = controls.get("GLOBAL", {})
                    symbol_ctrl = controls.get(symbol, {})
                    global_enabled = global_ctrl.get("enabled", True)
                    symbol_enabled = symbol_ctrl.get("enabled", True)
                    flatten_flag = bool(global_ctrl.get("flatten_on_disable")) or bool(
                        symbol_ctrl.get("flatten_on_disable")
                    )
                    blocked = (not global_enabled) or (not symbol_enabled)

                    if blocked:
                        if position and flatten_flag:
                            close_position(cur, position, price, reason="control_flatten")
                            position = _bot.load_position(cur, symbol)
                        elif position:
                            manage_position(cur, position, price)
                            position = _bot.load_position(cur, symbol)

                        update_asset(
                            symbol=symbol,
                            regime="paused",
                            strategy="kill_switch",
                            signal=None,
                            position=_build_position_state(position),
                        )
                        continue

                    if position:
                        manage_position(cur, position, price)
                        position = _bot.load_position(cur, symbol)

                    df = _bot.fetch_historical_data(symbol)
                    if df.empty:
                        update_asset(
                            symbol=symbol,
                            regime="unknown",
                            strategy="data_unavailable",
                            signal=None,
                            position=_build_position_state(position),
                        )
                        continue

                    signal = _bot.generate_signal(symbol, df)
                    if signal and signal.strategy != "no_trade":
                        print(
                            f"[SIGNAL] {symbol} | {signal.strategy} | conf={signal.confidence:.2f}",
                            flush=True,
                        )

                    update_asset(
                        symbol=symbol,
                        regime=signal.regime if signal else "unknown",
                        strategy=signal.strategy if signal else "none",
                        signal={
                            "side": signal.side if signal else None,
                            "confidence": getattr(signal, "confidence", None),
                        }
                        if signal
                        else None,
                        position=_build_position_state(position),
                    )

                    if (
                        signal
                        and signal.side == "LONG"
                        and signal.strategy != "no_trade"
                        and not position
                    ):
                        cur.execute("SELECT COUNT(*) FROM positions")
                        active_trades = int(cur.fetchone()[0] or 0)
                        if active_trades >= _bot.MAX_POSITIONS:
                            continue

                        now = time.time()
                        if symbol in last_trade_time and now - last_trade_time[symbol] < _bot.MAX_COOLDOWN_SECONDS:
                            continue

                        strategy_mult = get_strategy_multiplier(cur, signal.strategy, signal.regime)
                        size_multiplier = max(
                            0.0,
                            float(getattr(signal, "size_multiplier", 1.0) or 1.0),
                        )
                        size, deployed = calculate_position(
                            symbol=symbol,
                            price=price,
                            total_cap=total_cap,
                            stop_loss_pct=signal.stop_loss_pct,
                            confidence=signal.confidence,
                            regime_multiplier=strategy_mult,
                            size_multiplier=size_multiplier,
                        )

                        if size and size > 0:
                            open_position(
                                cur=cur,
                                symbol=symbol,
                                price=price,
                                size=size,
                                deployed_capital=deployed,
                                direction=signal.side,
                                regime=signal.regime,
                                strategy=signal.strategy,
                                stop_loss_pct=signal.stop_loss_pct,
                                take_profit_pct=signal.take_profit_pct,
                                secondary_take_profit_pct=signal.secondary_take_profit_pct,
                                tp3_pct=signal.tp3_pct,
                                tp3_close_fraction=signal.tp3_close_fraction,
                                trail_pct=signal.trail_pct,
                                tp1_close_fraction=signal.tp1_close_fraction,
                                tp2_close_fraction=signal.tp2_close_fraction,
                                confidence=signal.confidence,
                            )
                            print(f"[ENTRY] {symbol}", flush=True)
                            last_trade_time[symbol] = now

                conn.commit()

                try:
                    state = get_state()
                    if state.get("assets"):
                        if not _bot.push_to_caffeine(state):
                            print("[CAFFEINE PUSH] Not delivered", flush=True)
                except Exception as exc:
                    print(f"[CAFFEINE ERROR] {exc}", flush=True)

            except Exception as exc:
                if conn:
                    conn.rollback()
                print(f"[CRITICAL ERROR] {exc}", flush=True)
            finally:
                if cur:
                    cur.close()
                if conn:
                    conn.close()

            time.sleep(3)

    _bot.run_bot = _patched_run_bot
