from __future__ import annotations
import argparse,json,time
import ccxt
import numpy as np,pandas as pd
from strategy import compute_indicators,generate_signal,StrategyState
exchange=ccxt.binance({"enableRateLimit":True,"timeout":20000})
TAKER_FEE_BPS=6.0;MAKER_FEE_BPS=2.0;SLIPPAGE_BPS=3.0;SLIPPAGE_ATR_MULT=0.1

def _to_ms(v):
    if not v:return None
    ts=pd.Timestamp(v);ts=ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp()*1000)

def _sig(s,k,d=None):return getattr(s,k,d) if s is not None else d

def _slip(p,atr,c,side):
    atr_pct=(atr/c) if c else 0;sl=(SLIPPAGE_BPS/10000)+(atr_pct*SLIPPAGE_ATR_MULT)
    return p*(1+sl) if side=="LONG" else p*(1-sl)

def fetch_ohlcv_full(sym,tf,since=None,until=None):
    rows=[];cur=since
    while True:
        chunk=exchange.fetch_ohlcv(sym,timeframe=tf,since=cur,limit=1000)
        if not chunk:break
        rows.extend(chunk);cur=chunk[-1][0]+1
        if len(chunk)<1000 or (until and cur>=until):break
        time.sleep(exchange.rateLimit/1000)
    df=pd.DataFrame(rows,columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True)
    return compute_indicators(df.reset_index(drop=True))

def run_backtest(sym,tf,start=None,end=None):
    df=fetch_ohlcv_full(sym,tf,_to_ms(start),_to_ms(end))
    cap=10000;cash=cap;pos=None;trades=[];eq=[];cool=-1;state=StrategyState()
    for i in range(80,len(df)-1):
        w=df.iloc[:i+1];bar=df.iloc[i+1];idx=i+1
        if pos:
            hit_sl=bar["low"]<=pos["sl"];hit_tp=bar["high"]>=pos["tp"]
            if hit_sl or hit_tp:
                ex=pos["sl"] if hit_sl else pos["tp"];ex=_slip(ex,bar["atr"],bar["close"],pos["side"]) 
                fee=ex*pos["qty"]*(MAKER_FEE_BPS/10000);pnl=(ex-pos["entry"])*pos["qty"]-fee
                cash+=pos["qty"]*ex;trades.append(pnl);cool=idx+pos.get("cooldown",0);pos=None
        sig=generate_signal(w,state=state,symbol=sym)
        if pos is None and sig and idx>=cool:
            side=_sig(sig,"side");ep=_slip(bar["open"],bar["atr"],bar["close"],side)
            qty=(cash*0.3)/ep;fee=ep*qty*(TAKER_FEE_BPS/10000)
            sl=ep*(1-_sig(sig,"stop_loss_pct"));tp=ep*(1+_sig(sig,"take_profit_pct"))
            pos={"entry":ep,"qty":qty,"sl":sl,"tp":tp,"side":side,"cooldown":_sig(sig,"cooldown_bars",0)}
            cash-=qty*ep+fee
        eq.append(cash+(pos["qty"]*bar["close"] if pos else 0))
    return {"trades":len(trades),"profit_factor":sum([t for t in trades if t>0])/abs(sum([t for t in trades if t<0]) or 1),"final_equity":cash,"return_pct":(cash/cap-1)*100,"max_drawdown_pct":float((np.array(eq)-np.maximum.accumulate(eq)).min()/cap*100)}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--symbol",default="BTC/USDT");ap.add_argument("--timeframe",default="1h");ap.add_argument("--start");ap.add_argument("--end");a=ap.parse_args()
    print(json.dumps(run_backtest(a.symbol,a.timeframe,a.start,a.end),indent=2))
