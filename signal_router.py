# signal_router.py

def generate_signal(symbol, df_ltf, df_htf, state, controls):
    mode = get_mode_for_symbol(symbol)
    cfg = MODE_CONFIG[mode]

    if mode == "trend":
        return generate_signal_trend_btc(symbol, df_ltf, df_htf, state, controls, cfg)
    return generate_signal_reclaim_alt(symbol, df_ltf, df_htf, state, controls, cfg)
