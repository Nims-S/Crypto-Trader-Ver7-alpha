"""
main.py — Flask web server + bot thread launcher.
"""

import os
import re
import threading
import time
import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

from bot import run_bot
from config import PORT, BOT_VERSION, RESET_TOKEN
from db import get_conn, init_db
from price_feed import feeds
from risk import get_dynamic_capital
from state import get_controls, get_state, set_control
from api_v2 import api_v2
app = Flask(__name__)
app.register_blueprint(api_v2, url_prefix="/api/v2")
# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://miner-bot-epc.caffeine.xyz,https://caffeine.ai",
    ).split(",")
    if o.strip()
]

CAFFEINE_ORIGIN_REGEX = re.compile(r"^https://[a-zA-Z0-9-]+\.caffeine\.xyz$")

CORS(
    app,
    resources={
        r"/caffeine/*": {"origins": ALLOWED_ORIGINS + [CAFFEINE_ORIGIN_REGEX]},
        r"/*":          {"origins": ALLOWED_ORIGINS},
    },
    supports_credentials=False,
)


class IgnoreCaffeineFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "/caffeine/" in msg:
            return "UptimeRobot" in msg
        return True


logging.getLogger("werkzeug").addFilter(IgnoreCaffeineFilter())


BOT_THREAD_LOCK    = threading.Lock()
BOT_THREAD_STARTED = False
BOT_THREAD_ENABLED = os.getenv("BOT_THREAD_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on"
}

_LOCK_CONN = None


def background_executor():
    global _LOCK_CONN
    print("⏳ Waiting for web server to stabilise...", flush=True)
    time.sleep(5)

    try:
        _LOCK_CONN = get_conn()
        cur = _LOCK_CONN.cursor()
        cur.execute("SELECT pg_try_advisory_lock(12345)")

        if not cur.fetchone()[0]:
            print("⚠️ Another bot instance is already running. Skipping.", flush=True)
            return

        print("🗄️ Initialising database...", flush=True)
        init_db()

        print("🤖 Starting bot loop...", flush=True)
        run_bot()

    except Exception as e:
        print(f"❌ CRITICAL BACKGROUND ERROR: {e}", flush=True)


def start_background_executor_once():
    global BOT_THREAD_STARTED

    if not BOT_THREAD_ENABLED:
        return

    with BOT_THREAD_LOCK:
        if BOT_THREAD_STARTED:
            return

        thread = threading.Thread(target=background_executor, daemon=True)
        thread.start()
        BOT_THREAD_STARTED = True


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({"status": "running", "engine": f"algobot_{BOT_VERSION}"})


@app.route("/caffeine/state")
def caffeine_state():
    state = get_state()
    state["schema_version"] = 1
    return jsonify(state)


@app.route("/caffeine/controls", methods=["GET"])
def caffeine_controls():
    return jsonify(get_controls())


@app.route("/caffeine/controls", methods=["POST"])
def caffeine_controls_update():
    data = request.get_json(force=True) or {}
    scope = data.get("scope", "GLOBAL")
    enabled = data.get("enabled")
    flatten_on_disable = data.get("flatten_on_disable")

    if enabled is None and flatten_on_disable is None:
        return jsonify({"error": "nothing to update"}), 400

    snapshot = set_control(scope=scope, enabled=enabled, flatten_on_disable=flatten_on_disable)
    return jsonify({"ok": True, "controls": snapshot, "schema_version": 1})


@app.route("/health")
def health():
    return jsonify({"status": "alive", "version": BOT_VERSION})

@app.route("/sync_levels", methods=["POST"])
def sync_levels():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT symbol FROM positions")
    symbols = [r[0] for r in cur.fetchall()]

    for symbol in symbols:
        # fetch position + call update_position_levels
        pass

    conn.commit()
    return {"status": "synced"}
    
@app.route("/reset", methods=["POST"])
def reset():
    if RESET_TOKEN:
        token = request.args.get("token") or request.headers.get("X-Reset-Token")
        if token != RESET_TOKEN:
            return jsonify({"error": "unauthorized"}), 403

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM positions")
    cur.execute("DELETE FROM trades")
    cur.execute("DELETE FROM strategy_stats")

    conn.commit()
    conn.close()

    return jsonify({"status": "reset done"})


# ── startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_background_executor_once()
    port = int(os.environ.get("PORT", PORT))
    app.run(host="0.0.0.0", port=port)
else:
    start_background_executor_once()
