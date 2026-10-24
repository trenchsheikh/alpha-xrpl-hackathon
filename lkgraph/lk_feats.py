"""v4: group features on capped greedy clusters + LGBM (grouped folds, target=Hard) . usage: lk_v4.py"""
import pandas as pd,numpy as np,sys;
from lk_greedy import greedy
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
def feats(t,P,p):
  c=greedy(len(t),P,p,15,.5);t=t.assign(c=c);same=c[P.ia.values]==c[P.ib.values]
  mp=pd.Series(p[same]).groupby(c[P.ia.values][same]).agg(['mean','min','size'])
  rows={}
  for cc,g in t.sort_values('ts').groupby('c'):
    if len(g)<4: continue
    mids=set(g.ACCOUNT)&set(g.dst);bal={};unc=tot=0;lags=[];full=0
    last_in={}
    for a,b,u,ts in zip(g.ACCOUNT,g.dst,g.u,g.ts):
      if a in mids:
        tot+=1;unc+=bal.get(a,0)<u*.999
        if a in last_in: lags.append(ts-last_in[a])
      bal[b]=bal.get(b,0)+u;bal[a]=bal.get(a,0)-u;last_in[b]=ts
    full=sum(1 for m in mids if bal.get(m,0)>=-1e-6)
    gi=g[g.dst.isin(mids)].u.sum();go=g[g.ACCOUNT.isin(mids)].u.sum()
    rows[cc]=dict(g_n=len(g),g_cov=1-unc/tot if tot else -1,g_nmid=len(mids),g_full=full,g_span=g.ts.max()-g.ts.min(),
      g_ndays=(g.ts//24).nunique(),g_lag_med=np.median(lags) if lags else -1,g_lag_min=min(lags) if lags else -1,
      g_cons=go/gi if gi>0 else -1,g_nw=len(set(g.ACCOUNT)|set(g.dst)),g_gapmed=np.median(np.diff(g.ts.values)),g_ucv=g.u.std()/g.u.mean())
  G=pd.DataFrame.from_dict(rows,orient='index').join(mp.add_prefix('p_'))
  X=t[['c']].join(G,on='c').drop(columns='c');X['u']=np.log1p(t.u.values)
  return X
