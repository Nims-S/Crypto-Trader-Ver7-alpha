# api_v2.py

import time
import json
from flask import Blueprint, request, jsonify, Response

from db import get_conn
from config import CAPITAL
from state import get_controls

api_v2 = Blueprint("api_v2", __name__)

# ── simple in-memory cache ─────────────────────────────────────────────
CACHE = {}
CACHE_TTL = 5


def cached(key, fn):
    now = time.time()
    if key in CACHE:
        val, ts = CACHE[key]
        if now - ts < CACHE_TTL:
            return val

    val = fn()
    CACHE[key] = (val, now)
    return val


# ── overview (dashboard) ───────────────────────────────────────────────
@api_v2.route("/overview")
def overview():
    def compute():
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT COALESCE(SUM(pnl),0) FROM trades")
        total_pnl = float(cur.fetchone()[0] or 0)

        cur.execute("SELECT COUNT(*) FROM positions")
        open_positions = int(cur.fetchone()[0] or 0)

        cur.execute("""
            SELECT COALESCE(SUM(pnl),0)
            FROM trades
            WHERE timestamp >= NOW() - INTERVAL '1 day'
        """)
        daily_pnl = float(cur.fetchone()[0] or 0)

        conn.close()

        return {
            "equity": CAPITAL + total_pnl,
            "total_pnl": total_pnl,
            "daily_pnl": daily_pnl,
            "open_positions": open_positions,
            "last_update": time.time()
        }

    return jsonify(cached("overview", compute))


# ── positions (assets page) ────────────────────────────────────────────
@api_v2.route("/positions")
def positions():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT symbol, entry, sl, tp, tp2, tp3, size, strategy
        FROM positions
    """)

    rows = cur.fetchall()
    conn.close()

    data = [
        {
            "symbol": r[0],
            "entry": r[1],
            "sl": r[2],
            "tp1": r[3],
            "tp2": r[4],
            "tp3": r[5],
            "size": float(r[6]),
            "strategy": r[7],
        }
        for r in rows
    ]

    return jsonify(data)


# ── trades (paginated) ─────────────────────────────────────────────────
@api_v2.route("/trades")
def trades():
    page = int(request.args.get("page", 1))
    limit = min(int(request.args.get("limit", 20)), 100)
    offset = (page - 1) * limit

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM trades")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT symbol, entry, exit, pnl, timestamp, strategy
        FROM trades
        ORDER BY timestamp DESC
        LIMIT %s OFFSET %s
    """, (limit, offset))

    rows = cur.fetchall()
    conn.close()

    data = [
        {
            "symbol": r[0],
            "entry": r[1],
            "exit": r[2],
            "pnl": r[3],
            "timestamp": r[4].isoformat(),
            "strategy": r[5],
        }
        for r in rows
    ]

    return jsonify({
        "page": page,
        "limit": limit,
        "total": total,
        "data": data
    })


# ── controls (reuse existing logic) ────────────────────────────────────
@api_v2.route("/controls")
def controls():
    return jsonify(get_controls())


# ── streaming (SSE) ────────────────────────────────────────────────────
@api_v2.route("/stream")
def stream():
    def event_stream():
        while True:
            try:
                conn = get_conn()
                cur = conn.cursor()

                cur.execute("SELECT COUNT(*) FROM positions")
                positions = cur.fetchone()[0]

                cur.execute("SELECT COALESCE(SUM(pnl),0) FROM trades")
                pnl = cur.fetchone()[0]

                conn.close()

                payload = {
                    "positions": positions,
                    "pnl": pnl,
                    "ts": time.time()
                }

                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(2)

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                time.sleep(5)

    return Response(event_stream(), mimetype="text/event-stream")