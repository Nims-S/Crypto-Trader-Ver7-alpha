import os

BOT_VERSION = "v6 beta"

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

CAPITAL = float(os.getenv("CAPITAL", 100000))
RISK = float(os.getenv("RISK", 0.01))  # max risk per trade as fraction of equity
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 3))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", 0.03))
MAX_WEEKLY_LOSS_PCT = float(os.getenv("MAX_WEEKLY_LOSS_PCT", 0.06))
MAX_SYMBOL_EXPOSURE_PCT = float(os.getenv("MAX_SYMBOL_EXPOSURE_PCT", 0.40))
MAX_COOLDOWN_SECONDS = int(os.getenv("MAX_COOLDOWN_SECONDS", 900))
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1m")
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", 200))

# Hard regime gate for live trading. Start with trend-only, then widen later.
ALLOWED_REGIMES = {
    r.strip().lower()
    for r in os.getenv("ALLOWED_REGIMES", "trend").split(",")
    if r.strip()
}

# Allocation caps keep the bot from overconcentrating even when volatility is low.
ALLOCATION = {
    "BTC/USDT": float(os.getenv("BTC_ALLOCATION", 0.40)),
    "ETH/USDT": float(os.getenv("ETH_ALLOCATION", 0.30)),
    "SOL/USDT": float(os.getenv("SOL_ALLOCATION", 0.30)),
}

DB_URL = os.getenv("DATABASE_URL")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Optional safety token for destructive operations like reset
RESET_TOKEN = os.getenv("RESET_TOKEN")

PORT = int(os.getenv("PORT", 10000))
