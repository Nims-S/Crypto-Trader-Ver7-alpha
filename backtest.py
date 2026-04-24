# (only showing modified section)

    df = fetch_ohlcv_full(sym, tf, _to_ms(start), _to_ms(end))
    if df.empty:
        return {"error": f"no data returned for {sym} on {tf}"}

    # indicators already computed in fetch_ohlcv_full → no recompute in loop

    htf_tf = _htf_timeframe_for_symbol(sym, tf)
    df_htf = fetch_ohlcv_full(sym, htf_tf, _to_ms(start), _to_ms(end))

    # Pre-align HTF index for fast lookup
    df_htf = df_htf.sort_index()

    cap = 10_000.0

    # --- rest unchanged except signal call ---

            # FAST HTF slice (timestamp-based instead of growing iloc)
            htf_slice = df_htf.loc[:bar.name]

            sig = generate_signal(w, state=state, symbol=sym, df_htf=htf_slice)
