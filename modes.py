# modes.py (or near the top of backtest.py)

BTC_SYMBOLS = {"BTC/USDT"}
ALT_SYMBOLS = {"ETH/USDT", "BNB/USDT", "LINK/USDT", "AVAX/USDT", "SOL/USDT"}

MODE_BY_SYMBOL = {
    **{s: "trend" for s in BTC_SYMBOLS},
    **{s: "mean_reversion" for s in ALT_SYMBOLS},
}

MODE_CONFIG = {
    "trend": {
        "htf_filter": True,
        "displacement_threshold": 1.2,
        "sweep_depth": 0.0,          # not used in trend mode
        "tp1_rr": 1.5,
        "tp2_rr": 3.5,
        "be_trigger_rr": 1.0,
        "max_bars_in_trade": 80,
        "runner_enabled": True,
        "partial_tp": True,
    },
    "mean_reversion": {
        "htf_filter": True,
        "displacement_threshold": 0.8,
        "sweep_depth": 0.25,
        "tp1_rr": 0.8,
        "tp2_rr": 1.4,
        "be_trigger_rr": 0.6,
        "max_bars_in_trade": 18,
        "runner_enabled": False,
        "partial_tp": True,
    },
}


def get_mode_for_symbol(symbol: str) -> str:
    return MODE_BY_SYMBOL.get(symbol, "mean_reversion")
