import os
import requests

CAFFEINE_URL = os.getenv("CAFFEINE_URL")
CAFFEINE_TOKEN = os.getenv("CAFFEINE_TOKEN")


def push_to_caffeine(data):
    """Push bot state to the external Caffeine dashboard.

    Returns True on a successful 2xx response, False otherwise.
    """
    if not CAFFEINE_URL:
        print("[CAFFEINE PUSH] CAFFEINE_URL not set.", flush=True)
        return False

    headers = {
        "Content-Type": "application/json",
    }
    if CAFFEINE_TOKEN:
        headers["Authorization"] = f"Bearer {CAFFEINE_TOKEN}"

    try:
        response = requests.post(
            CAFFEINE_URL,
            json=data,
            headers=headers,
            timeout=3,
        )
        if response.status_code >= 400:
            print(
                f"[CAFFEINE PUSH ERROR] HTTP {response.status_code} from {CAFFEINE_URL}: {response.text[:200]}",
                flush=True,
            )
            return False
        return True
    except Exception as e:
        print(f"[CAFFEINE PUSH ERROR] {e}", flush=True)
        return False
