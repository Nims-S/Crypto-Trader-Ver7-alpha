import threading
import requests
from config import BOT_TOKEN, CHAT_ID


def send_telegram(msg: str) -> None:
    """Fire-and-forget Telegram notification.

    Runs in a daemon thread so it never blocks the bot loop, even if the
    Telegram API is slow or unreachable.
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("[TELEGRAM] BOT_TOKEN or CHAT_ID not set.", flush=True)
        return
    thread = threading.Thread(target=_send, args=(msg,), daemon=True)
    thread.start()


def _send(msg: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(
                f"[TELEGRAM ERROR] {response.status_code}: {response.text}",
                flush=True,
            )
    except Exception as e:
        print(f"[TELEGRAM CRITICAL] {e}", flush=True)
