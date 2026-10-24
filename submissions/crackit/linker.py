"""Learned pair linker -> clusters -> covered-send score.
usage: linker.py   (writes /workspace/crackit/lk_train.csv, lk_test.csv)"""
import pandas as pd,numpy as np,json,sys
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
MINU=30;MAXDT=150
def load(D):
  t=pd.read_parquet(D+'/transactions.parquet');t=t[t.TRANSACTION_TYPE=='Payment']
  t['dst']=t.EXTRA_TRANSACTION_FIELDS.map(lambda s:json.loads(s).get('Destination'))
  tr=pd.read_parquet(D+'/transfers.parquet',columns=['TRANSACTION_HASH','SYMBOL','VALUE']).groupby('TRANSACTION_HASH').last()
  mr=pd.read_csv(D+'/market_rates.csv');mr['sym']=mr.ASSET.str.split(':').str[-1];px=mr.groupby('sym').PRICE_MICRO_USD.median()/1e6
  t['sym']=t.TRANSACTION_HASH.map(tr.SYMBOL).fillna('?');t['u']=t.TRANSACTION_HASH.map(tr.VALUE*tr.SYMBOL.map(px)).fillna(0)
  t['ts']=t.TIMESTAMP.astype(float)/3600;t=t[t.ACCOUNT!=t.dst].reset_index(drop=True)
  return t
def pairs(t):
  deg=pd.concat([t.ACCOUNT,t.dst]).value_counts()
  E=pd.concat([pd.DataFrame({'i':t.index,'w':t.ACCOUNT+'|'+t.sym,'cp':t.dst,'dir':1,'ts':t.ts,'u':t.u}),
               pd.DataFrame({'i':t.index,'w':t.dst+'|'+t.sym,'cp':t.ACCOUNT,'dir':0,'ts':t.ts,'u':t.u})],ignore_index=True)
  E=E[E.u>=MINU].sort_values(['w','ts']).reset_index(drop=True)
  E['k']=E.groupby('w').cumcount();E['wn']=E.groupby('w').i.transform('size')
  E['cpdeg']=E.cp.str.split('|').str[0].map(deg).fillna(0)
  P=E.merge(E,on='w',suffixes=('a','b'));P=P[(P.ka<P.kb)];P=P[(P.tsb-P.tsa)<MAXDT]
  f=pd.DataFrame({'dt':P.tsb-P.tsa,'dk':P.kb-P.ka,'da':P.dira,'db':P.dirb,'lr':np.log(P.ub/P.ua),'lua':np.log(P.ua),'lub':np.log(P.ub),
    'samecp':(P.cpa==P.cpb).astype(int),'hod':((P.tsa%24)-(P.tsb%24)).abs(),'dday':(P.tsb//24-P.tsa//24),
    'cpda':P.cpdega,'cpdb':P.cpdegb,'wn':P.wna})
  # repeated pair counts
  pc=E.groupby(['w','cp']).size();f['repa']=pc.reindex(list(zip(P.w,P.cpa))).values;f['repb']=pc.reindex(list(zip(P.w,P.cpb))).values
  # local density: txs at w within 2h of each
  return P[['ia','ib']].reset_index(drop=True),f.reset_index(drop=True)
def score(t,P,p,thr):
  m=p>thr;n=len(t);A=coo_matrix((np.ones(m.sum()),(P.ia[m],P.ib[m])),shape=(n,n))
  _,c=connected_components(A,directed=False);t=t.assign(c=c).sort_values('ts')
  sz=t.c.value_counts();res={}
  for cc,g in t[t.c.map(sz).between(6,80)].groupby('c'):
    mids=set(g.ACCOUNT)&set(g.dst)
    if not mids: continue
    bal={};unc=tot=0
    for a,b,u in zip(g.ACCOUNT,g.dst,g.u):
      if a in mids:
        tot+=1;unc+=bal.get(a,0)<u*.999
      bal[b]=bal.get(b,0)+u;bal[a]=bal.get(a,0)-u
    res[cc]=1-unc/tot
  return t.c.map(res).fillna(-1).sort_index().values,t.c.sort_index().values
if __name__=='__main__':
  tr=load('/workspace/xrpl-data/train');te=load('/workspace/xrpl-data/test')
  l=pd.read_csv('/workspace/xrpl-data/train/labels.csv');tr=tr.merge(l,on='TRANSACTION_HASH',how='left')
  g=np.where(tr.cohort_id.notna(),'c:'+tr.cohort_id.astype(str),np.where(tr.scenario_id.notna(),'s:'+tr.scenario_id.astype(str),'t:'+tr.TRANSACTION_HASH))
  # folds on full labels file to match grouped_cv
  L=l.copy();gl=np.where(L.cohort_id.notna(),'c:'+L.cohort_id.astype(str),np.where(L.scenario_id.notna(),'s:'+L.scenario_id.astype(str),'t:'+L.TRANSACTION_HASH))
  fold=np.zeros(len(L),int)
  for k,(_,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(L,L.is_suspicious,gl)):fold[va]=k
  tr['fold']=tr.TRANSACTION_HASH.map(dict(zip(L.TRANSACTION_HASH,fold)))
  P,F=pairs(tr);sc=tr.scenario_id.values
  y=((sc[P.ia]==sc[P.ib])&pd.notna(sc[P.ia])).astype(int);fa=tr.fold.values[P.ia];fb=tr.fold.values[P.ib]
  print('pairs',len(P),'pos',y.sum(),flush=True)
  p=np.zeros(len(P));cnt=np.zeros(len(P))
  ms=[]
  for k in range(5):
    trn=(fa!=k)&(fb!=k)
    m=LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=63,subsample=.5,subsample_freq=1,n_jobs=8,verbose=-1,random_state=42).fit(F[trn],y[trn])
    ms.append(m);hit=(fa==k)|(fb==k);p[hit]+=m.predict_proba(F[hit])[:,1];cnt[hit]+=1
  p/=np.maximum(cnt,1)
  from sklearn.metrics import roc_auc_score,average_precision_score
  print('pair AUC',roc_auc_score(y,p),'AP',average_precision_score(y,p),flush=True)
  np.save('/tmp/lk_p.npy',p);P.to_pickle('/tmp/lk_P.pkl');tr.to_pickle('/tmp/lk_tr.pkl')
  Pt,Ft=pairs(te);pt=np.mean([m.predict_proba(Ft)[:,1] for m in ms],0);np.save('/tmp/lk_pt.npy',pt);Pt.to_pickle('/tmp/lk_Pt.pkl');te.to_pickle('/tmp/lk_te.pkl')
