<FULL FILE OMITTED FOR BREVITY WITH PATCH>

# PATCH ADDITION inside run_backtest():
    since = _to_ms(start)
    until = _to_ms(end)

    # ✅ FIX: clamp future timestamps
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    if until:
        until = min(until, now_ms)

    df = fetch_ohlcv_full(sym, tf, since, until, use_cache=use_cache)

# (rest of file unchanged)
