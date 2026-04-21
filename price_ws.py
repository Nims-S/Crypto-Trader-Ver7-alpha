import websocket
import json
import threading
import time
import requests


class PriceFeedManager:
    def __init__(self, symbol):
        self.symbol = symbol.lower()
        self.price = None
        self.ws = None
        self.running = False
        self.last_update = 0

    def get_price(self):
        # refresh via REST if stale
        if time.time() - self.last_update > 5:
            self._fetch_rest_price()
        return self.price

    def age_seconds(self):
        if not self.last_update:
            return None
        return time.time() - self.last_update

    def is_stale(self, max_age_seconds: int = 20):
        age = self.age_seconds()
        return age is not None and age > max_age_seconds

    def _fetch_rest_price(self):
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={self.symbol.upper()}"
            res = requests.get(url, timeout=2)
            if res.status_code == 200:
                self.price = float(res.json()["price"])
                self.last_update = time.time()
        except Exception:
            pass

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.price = float(data["c"])
            self.last_update = time.time()
        except Exception as e:
            print(f"[WS PARSE ERROR] {self.symbol}: {e}", flush=True)

    def _on_error(self, ws, error):
        print(f"[WS ERROR] {self.symbol}: {error}", flush=True)

    def _on_close(self, ws, close_status_code, close_msg):
        self.running = False

    def _on_open(self, ws):
        print(f"[WS CONNECTED] {self.symbol}", flush=True)

    def _run(self):
        while True:
            try:
                url = f"wss://stream.binance.com:9443/ws/{self.symbol}@ticker"
                self.ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                self.running = True
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[WS CRASH] {self.symbol}: {e}", flush=True)
            time.sleep(5)

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
