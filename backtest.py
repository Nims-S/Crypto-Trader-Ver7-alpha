from __future__ import annotations
import argparse, json, time
import ccxt
import numpy as np, pandas as pd
from strategy import compute_indicators, generate_signal, StrategyState

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})

TAKER_FEE_BPS  = 6.0
MAKER_FEE_BPS  = 2.0
SLIPPAGE_BPS   = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE = 0.01   # 1% of equity per trade (replaces flat 30% notional)
MAX_NOTIONAL_FRAC = 0.25  # hard cap: never more than 25% of equity in one trade


def _to_ms(v):
    if not v:
        return None
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _sig(s, k, d=None):
    return getattr(s, k, d) if s is not None else d


def _slip(p, atr, c, side):
    atr_pct = (atr / c) if c else 0
    sl = (SLIPPAGE_BPS / 10000) + (atr_pct * SLIPPAGE_ATR_MULT)
    return p * (1 + sl) if side == "LONG" else p * (1 - sl)


def fetch_ohlcv_full(sym, tf, since=None, until=None):
    rows = []
    cur  = since
    while True:
        chunk = exchange.fetch_ohlcv(sym, timeframe=tf, since=cur, limit=1000)
        if not chunk:
            break
        rows.extend(chunk)
        cur = chunk[-1][0] + 1
        if len(chunk) < 1000 or (until and cur >= until):
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Precompute indicators ONCE on the full df (not per-bar)
    df = compute_indicators(df.reset_index())

    # ── 200 EMA for regime filter: computed causal (expanding, no future leak) ──
    # We use .ewm with min_periods so early bars naturally have less data.
    # This is stored per-row so we can look it up O(1) in the loop.
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df["ema200"] = df["close"].ewm(span=200, min_periods=1).mean()

    return df


def run_backtest(sym, tf, start=None, end=None):
    df = fetch_ohlcv_full(sym, tf, _to_ms(start), _to_ms(end))

    cap   = 10_000.0
    cash  = cap
    pos   = None
    trades = []
    eq    = []
    cool  = -1
    state = StrategyState()

    for i in range(200, len(df) - 1):
        bar = df.iloc[i + 1]
        idx = i + 1

        # Current mark-to-market equity (used for risk sizing)
        equity = cash + (pos["qty"] * df.iloc[i]["close"] if pos else 0.0)

        # ── Manage open trade ────────────────────────────────────────────────
        if pos:
            hit_sl = bar["low"]  <= pos["sl"]
            hit_tp = bar["high"] >= pos["tp"]
            if hit_sl or hit_tp:
                ex  = pos["sl"] if hit_sl else pos["tp"]
                ex  = _slip(ex, bar["atr"], bar["close"], pos["side"])
                fee = ex * pos["qty"] * (MAKER_FEE_BPS / 10000)
                pnl = (ex - pos["entry"]) * pos["qty"] - fee
                cash += pos["qty"] * ex
                trades.append({
                    "ts":     bar.name,
                    "side":   pos["side"],
                    "entry":  round(pos["entry"], 2),
                    "exit":   round(ex, 2),
                    "pnl":    round(pnl, 4),
                    "result": "SL" if hit_sl else "TP",
                })
                cool = idx + pos.get("cooldown", 0)
                pos  = None

        # ── Look for new signal when flat ────────────────────────────────────
        if pos is None and idx >= cool:
            # Pass the window ending at bar i (the CLOSED bar), not bar i+1
            w   = df.iloc[:i + 1]
            sig = generate_signal(w, state=state, symbol=sym)

            if sig:
                side = _sig(sig, "side")
                ep   = _slip(bar["open"], bar["atr"], bar["close"], side)
                sl_p = _sig(sig, "stop_loss_pct",  0.0)
                tp_p = _sig(sig, "take_profit_pct", 0.0)

                if sl_p < 0.0005:   # degenerate SL (< 0.05%), skip
                    continue

                sl = ep * (1 - sl_p)
                tp = ep * (1 + tp_p)

                # ── REGIME FILTER: causal 200 EMA (no look-ahead) ────────────
                ema200 = float(df.iloc[i]["ema200"])   # value at signal bar, not next bar
                if side == "LONG"  and ep < ema200:
                    continue
                if side == "SHORT" and ep > ema200:
                    continue

                # ── RISK-BASED SIZING ────────────────────────────────────────
                risk_amount = equity * RISK_PER_TRADE
                qty = risk_amount / (sl_p * ep)
                qty = min(qty, (equity * MAX_NOTIONAL_FRAC) / ep)  # hard cap

                fee  = ep * qty * (TAKER_FEE_BPS / 10000)
                cost = qty * ep + fee
                if cost > cash:
                    continue

                pos = {
                    "entry":    ep,
                    "qty":      qty,
                    "sl":       sl,
                    "tp":       tp,
                    "side":     side,
                    "cooldown": _sig(sig, "cooldown_bars", 0),
                }
                cash -= cost

        # Mark-to-market equity snapshot (includes open position at close)
        eq.append(cash + (pos["qty"] * bar["close"] if pos else 0.0))

    # ── Close any open position at last bar's close (for accurate final equity) ──
    if pos:
        last = df.iloc[-1]
        ex   = _slip(last["close"], last["atr"], last["close"], pos["side"])
        fee  = ex * pos["qty"] * (MAKER_FEE_BPS / 10000)
        pnl  = (ex - pos["entry"]) * pos["qty"] - fee
        cash += pos["qty"] * ex
        trades.append({
            "ts": last.name, "side": pos["side"],
            "entry": round(pos["entry"], 2), "exit": round(ex, 2),
            "pnl": round(pnl, 4), "result": "EOD_CLOSE",
        })

    # ── Stats ────────────────────────────────────────────────────────────────
    pnls      = [t["pnl"] for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_los = abs(sum(p for p in pnls if p < 0)) or 1e-9
    wins      = sum(1 for p in pnls if p > 0)

    eq_arr = np.array(eq if eq else [cap])
    peak   = np.maximum.accumulate(eq_arr)
    dd_pct = float(((eq_arr - peak) / peak).min() * 100)

    return {
        "trades":           len(trades),
        "win_rate":         round(wins / max(len(pnls), 1), 3),
        "profit_factor":    round(gross_win / gross_los, 4),
        "final_equity":     round(cash, 2),          # cash after all closed trades
        "return_pct":       round((cash / cap - 1) * 100, 4),
        "max_drawdown_pct": round(dd_pct, 4),
        "avg_rr_realised":  round(
            (gross_win / wins) / (gross_los / max(len(pnls) - wins, 1)), 3
        ) if wins > 0 and len(pnls) > wins else 0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",    default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    print(json.dumps(run_backtest(a.symbol, a.timeframe, a.start, a.end), indent=2))
