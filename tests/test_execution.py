import execution

class FakeCursor:
    def __init__(self):
        self.positions={};self.trades=[];self._last=None
    def execute(self,sql,params=None):
        q=" ".join(sql.split()).lower();params=params or ()
        if "insert into positions" in q:
            (symbol,entry,sl,tp1,tp2,tp3,size,orig,reg,conf,dir_,strat,slp,tp1p,tp2p,tr, f1,f2,f3)=params
            self.positions[symbol]={"symbol":symbol,"entry":entry,"sl":sl,"tp":tp1,"tp2":tp2,"tp3":tp3,"size":size,"original_size":orig,"regime":reg,"confidence":conf,"direction":dir_,"strategy":strat,"stop_loss_pct":slp,"take_profit_pct":tp1p,"secondary_take_profit_pct":tp2p,"trail_pct":tr,"tp1_close_fraction":f1,"tp2_close_fraction":f2,"tp3_close_fraction":f3,"tp1_hit":False,"tp2_hit":False,"tp3_hit":False}
        elif "select size from positions" in q:
            s=self.positions.get(params[0]);self._last=(s["size"],) if s else None
        elif "select entry, direction, tp3" in q:
            s=self.positions.get(params[0]);self._last=(s["entry"],s["direction"],s.get("tp3")) if s else None
        elif "tp1_hit" in q and "update positions set size" in q:
            new_size,sym=params;self.positions[sym]["size"]=new_size;self.positions[sym]["tp1_hit"]=True
        elif "tp2_hit" in q and "update positions set size" in q:
            new_size,sym=params;self.positions[sym]["size"]=new_size;self.positions[sym]["tp2_hit"]=True
        elif "update positions set sl" in q:
            sl,sym=params;self.positions[sym]["sl"]=sl
        elif "insert into trades" in q:
            self.trades.append(params)
        elif "delete from positions" in q:
            self.positions.pop(params[0],None)
    def fetchone(self): return self._last


def _patch(monkeypatch):
    monkeypatch.setattr(execution,"send_telegram",lambda *a,**k:None)
    monkeypatch.setattr(execution,"log_trade_performance",lambda *a,**k:None)


def test_tp_cascade(monkeypatch):
    _patch(monkeypatch);cur=FakeCursor()
    execution.open_position(cur,"BTC/USDT",100,10,1000,"LONG","trend","test",0.01,0.05,0.1,0.15,0.0,0.0,0.5,0.3,0.2,0.8)
    pos=cur.positions["BTC/USDT"];execution.manage_position(cur,pos,price=110)
    assert "BTC/USDT" in cur.positions;assert len(cur.trades)>=1


def test_stop_after_partials(monkeypatch):
    _patch(monkeypatch);cur=FakeCursor()
    execution.open_position(cur,"BTC/USDT",100,10,1000,"LONG","trend","test",0.01,0.05,0.1,0.15,0.0,0.0,0.5,0.3,0.2,0.8)
    pos=cur.positions["BTC/USDT"];execution.manage_position(cur,pos,price=110)
    pos=cur.positions.get("BTC/USDT");execution.manage_position(cur,pos,price=90)
    assert "BTC/USDT" not in cur.positions
