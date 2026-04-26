from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from strategy import StrategyState, compute_indicators, generate_signal

exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
CACHE_DIR = Path(".backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TAKER_FEE_BPS = 6.0
MAKER_FEE_BPS = 2.0
SLIPPAGE_BPS = 3.0
SLIPPAGE_ATR_MULT = 0.1
RISK_PER_TRADE = 0.01
MAX_NOTIONAL_FRAC = 0.25

DEFAULT_TP1_R = 1.8
DEFAULT_TP2_R = 4.5
DEFAULT_TP1_QTY_FRAC = 0.20
DEFAULT_MOVE_BE_R = 1.8
DEFAULT_TRAIL_ATR_MULT = 1.5

MAX_BARS_BY_REGIME = {"trend": 30, "mean_reversion": 12}

REQUIRED_INDICATOR_COLS = {"atr","atr_pct","atr_pct_rank","bb_width","bb_width_rank","rolling_body","ema20","ema50","ema200","adx","rsi","macd_hist"}

@dataclass
class Position:
    side: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    be_trigger: float
    qty_open: float
    qty_tp1: float
    qty_tp2: float
    qty_tp3: float
    bars: int = 0
    max_bars: int = 72
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    be_moved: bool = False
    trail_pct: float = 0.0
    trail_atr_mult: float = DEFAULT_TRAIL_ATR_MULT
    size_multiplier: float = 1.0
    open_ts: str = ""

def _to_ms(v):
    if not v: return None
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp()*1000)

def _sig(s,k,d=None): return getattr(s,k,d) if s is not None else d

def _slip(p,atr,c,side):
    atr_pct=(atr/c) if c else 0.0
    sl=(SLIPPAGE_BPS/10000)+(atr_pct*SLIPPAGE_ATR_MULT)
    return p*(1+sl) if side=="LONG" else p*(1-sl)

def _pnl(entry,exit_p,qty,side,fee_bps):
    fee=exit_p*qty*(fee_bps/10000)
    return (exit_p-entry)*qty-fee if side=="LONG" else (entry-exit_p)*qty-fee

def _close_leg(cash,pos,exit_p,qty,result,trades):
    if qty<=0: return cash,pos
    qty=min(qty,pos.qty_open)
    pnl=_pnl(pos.entry,exit_p,qty,pos.side,MAKER_FEE_BPS)
    cash+=pos.entry*qty+pnl
    trades.append({"ts":pos.open_ts,"side":pos.side,"entry":round(pos.entry,2),"exit":round(exit_p,2),"qty":round(qty,6),"pnl":round(pnl,4),"result":result})
    pos.qty_open-=qty
    return (cash,None) if pos.qty_open<=1e-10 else (cash,pos)

def _trail_stop(pos,close,atr):
    if not pos.tp1_hit: return
    atr_mult=pos.trail_atr_mult or DEFAULT_TRAIL_ATR_MULT
    if pos.side=="LONG": pos.sl=max(pos.sl, close-(atr*atr_mult))
    else: pos.sl=min(pos.sl, close+(atr*atr_mult))

def _htf_timeframe_for_symbol(symbol,ltf):
    if ltf=="1d": return "1w"
    if ltf in {"15m","30m","1h","2h","4h"}: return "1d"
    return "1h"

def run_backtest(sym,tf,start=None,end=None,allow_shorts=False,max_bars=0,use_cache=True):
    since=_to_ms(start); until=_to_ms(end)
    now_ms=int(pd.Timestamp.now(tz="UTC").timestamp()*1000)
    if until: until=min(until,now_ms)

    df=fetch_ohlcv_full(sym,tf,since,until,use_cache)
    if df.empty: return {"error":"no data"}

    htf_tf=_htf_timeframe_for_symbol(sym,tf)
    df_htf=fetch_ohlcv_full(sym,htf_tf,since,until,use_cache)
    if df_htf.empty: return {"error":"no htf"}

    htf_pos=np.searchsorted(df_htf.index.values,df.index.values,side="right")-1

    cap=10000.0; cash=cap; pos=None; trades=[]; eq=[]; cool=-1
    state=StrategyState(allow_shorts=allow_shorts)

    for i in range(260,len(df)-1):
        bar=df.iloc[i+1]; idx=i+1
        atr=float(bar["atr"]); close=float(bar["close"])
        high=float(bar["high"]); low=float(bar["low"])

        if pos:
            pos.bars+=1
            if pos.bars>=pos.max_bars:
                cash,pos=_close_leg(cash,pos,_slip(close,atr,close,pos.side),pos.qty_open,"MAX",trades); cool=idx; eq.append(cash); continue

            sl_hit = low<=pos.sl if pos.side=="LONG" else high>=pos.sl
            if sl_hit:
                cash,pos=_close_leg(cash,pos,_slip(pos.sl,atr,close,pos.side),pos.qty_open,"SL",trades); cool=idx; eq.append(cash); continue

        if pos is None and idx>=cool:
            htf_slice=df_htf.iloc[:htf_pos[idx]+1] if htf_pos[idx]>=0 else df_htf.iloc[:0]
            sig=generate_signal(df.iloc[:i+1],state=state,symbol=sym,df_htf=htf_slice)
            if sig:
                ep=_slip(float(bar["open"]),atr,close,sig.side)
                sl=ep*(1-sig.stop_loss_pct)
                risk=abs(ep-sl)
                if risk<=0: continue
                qty=min((cash*0.01)/risk,(cash*0.25)/ep)
                cost=qty*ep
                if cost>cash: continue
                pos=Position(sig.side,ep,sl,ep+1.8*risk,ep+4.5*risk,ep+4.5*risk,ep+1.8*risk,qty,qty*0.2,qty*0.8,0)
                cash-=cost

        eq.append(cash+(pos.qty_open*close if pos else 0))

    return {"symbol":sym,"trades":len(trades),"final_equity":round(cash,2)}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol",default="BTC/USDT")
    ap.add_argument("--timeframe",default="1d")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a=ap.parse_args()
    print(json.dumps(run_backtest(a.symbol,a.timeframe,a.start,a.end),indent=2))