from price_ws import PriceFeedManager

feeds = {
    "BTC/USDT": PriceFeedManager("btcusdt"),
    "ETH/USDT": PriceFeedManager("ethusdt"),
    "SOL/USDT": PriceFeedManager("solusdt"),
}

for feed in feeds.values():
    feed.start()
