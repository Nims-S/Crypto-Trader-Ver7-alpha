from __future__ import annotations
import argparse,json,time
import ccxt
import numpy as np,pandas as pd
from strategy import compute_indicators,generate_signal,StrategyState
exchange=ccxt.binance({"enableRateLimit":True,"timeout":20000})
RISK=0.005;MAX_BARS=24

def _to_ms(v):
    if not v:return None
    ts=pd.Timestamp(v);ts=ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp()*1000)

def fetch(sym,tf,since=None,until=None):
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
    df=fetch(sym,tf,_to_ms(start),_to_ms(end))
    cap=10000;cash=cap;pos=None;trades=[];eq=[];state=StrategyState()
    for i in range(80,len(df)-1):
        w=df.iloc[:i+1];bar=df.iloc[i+1]
        if pos:
            pos['bars']+=1
            if (pos['side']=='LONG' and bar['low']<=pos['sl']) or (pos['side']=='SHORT' and bar['high']>=pos['sl']):
                pnl=(pos['sl']-pos['entry'])/pos['entry'] if pos['side']=='LONG' else (pos['entry']-pos['sl'])/pos['entry']
                cash*=1+pnl;trades.append(pnl);pos=None;continue
            if not pos['tp1_hit'] and ((pos['side']=='LONG' and bar['high']>=pos['tp1']) or (pos['side']=='SHORT' and bar['low']<=pos['tp1'])):
                pnl=(pos['tp1']-pos['entry'])/pos['entry'] if pos['side']=='LONG' else (pos['entry']-pos['tp1'])/pos['entry']
                cash*=1+pnl*0.5;pos['tp1_hit']=True;pos['sl']=pos['entry']
            if pos and ((pos['side']=='LONG' and bar['high']>=pos['tp2']) or (pos['side']=='SHORT' and bar['low']<=pos['tp2'])):
                pnl=(pos['tp2']-pos['entry'])/pos['entry'] if pos['side']=='LONG' else (pos['entry']-pos['tp2'])/pos['entry']
                cash*=1+pnl*0.5;trades.append(pnl);pos=None;continue
            if pos and pos['bars']>=MAX_BARS:
                pnl=(bar['close']-pos['entry'])/pos['entry'] if pos['side']=='LONG' else (pos['entry']-bar['close'])/pos['entry']
                cash*=1+pnl;trades.append(pnl);pos=None;continue
        sig=generate_signal(w,state=state,symbol=sym)
        if pos is None and sig:
            entry=bar['open'];sl=sig.stop_loss;dist=abs(entry-sl)
            tp1=entry+dist if sig.side=='LONG' else entry-dist
            tp2=entry+dist*2.5 if sig.side=='LONG' else entry-dist*2.5
            pos={'entry':entry,'sl':sl,'tp1':tp1,'tp2':tp2,'side':sig.side,'tp1_hit':False,'bars':0}
        eq.append(cash)
    pf=sum([t for t in trades if t>0])/abs(sum([t for t in trades if t<0]) or 1)
    return {"trades":len(trades),"profit_factor":pf,"final_equity":cash,"return_pct":(cash/cap-1)*100}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--symbol",default="BTC/USDT");ap.add_argument("--timeframe",default="1h");ap.add_argument("--start");ap.add_argument("--end");a=ap.parse_args()
    print(json.dumps(run_backtest(a.symbol,a.timeframe,a.start,a.end),indent=2))
