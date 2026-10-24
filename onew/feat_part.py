# Reconstruct relay cohorts (account-graph components w/o hubs, + 6h/24h windows); within-cohort relative feats; Hard-vs-rest LGBM; inject.
import pandas as pd,numpy as np,sys,zipfile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
R='/workspace/dev-tester/alpha-xrpl-hackathon/';sys.path.insert(0,R+'lkgraph');from inj import inject
l=pd.read_csv(R+'data/train/labels.csv');y=l.is_suspicious.values;hard=((l.difficulty=='hard')&(y==1)).values.astype(int)
q=pd.read_csv(R+'out/sub/h_graph3s-inject40.csv')
def feats(fm,fg,ft,ids):
  B=pd.read_parquet(fm).set_index('TRANSACTION_HASH').loc[ids]
  G=pd.read_parquet(fg).set_index('TRANSACTION_HASH').loc[ids]
  T=pd.read_parquet(ft,columns=['TRANSACTION_HASH','TIMESTAMP']).drop_duplicates('TRANSACTION_HASH').set_index('TRANSACTION_HASH').loc[ids].TIMESTAMP.values.astype(float)
  num=pd.concat([B.select_dtypes(exclude='object'),G.drop(columns=['ACCOUNT','dst'])],axis=1).reset_index(drop=True)
  acc=G.ACCOUNT.astype(str).values;dst=G.dst.astype(str).values;X=num.copy()
  u,inv=np.unique(np.r_[acc,dst],return_inverse=True);n=len(acc);a,d=inv[:n],inv[n:]
  deg=np.bincount(inv,minlength=len(u));ok=(deg[a]<60)&(deg[d]<60)
  m=coo_matrix((np.ones(ok.sum()),(a[ok],d[ok])),shape=(len(u),len(u)))
  _,cc=connected_components(m,directed=False);comp=np.where(ok,cc[a],-1-np.arange(n))
  key=[k for k in ['log_usd','relay_match6h','prior_in_delay','next_out_delay','prior_in_vs_next_out6h','relay_amount_matching','upstream_best_match_21600','downstream_best_match_21600','fundflow_continuity_4hop','s_lag_in','s_ratio_in','d_lag_out','d_ratio_out','hour','s_burst6','d_burst6'] if k in num]
  num['ts']=T;key2=key+['ts']
  coh={'C':comp,'C6':[f'{c}_{int(t//21600)}' for c,t in zip(comp,T)],'W6':(T//21600).astype(int),'S':acc,'D':dst}
  for who,gk in coh.items():
    gb=num[key2].groupby(gk);X[f'{who}_n']=num.groupby(gk).ts.transform('size').values
    X[[f'{who}rk_{k}' for k in key2]]=gb.rank(pct=True).values
    X[[f'{who}dz_{k}' for k in key2]]=(num[key2]-gb.transform('median')).values/(gb.transform('std').fillna(0).values+1e-9)
  X[[f'g_{k}' for k in key]]=num[key].rank(pct=True).values
  return X.drop(columns=['ts'],errors='ignore'),comp
Xtr,ctr=feats(R+'cache/feats_motif_train.parquet',R+'lkgraph/g_train.parquet',R+'data/train/transactions.parquet',l.TRANSACTION_HASH)
Xte,_=feats(R+'cache/test_feats_motif.parquet',R+'lkgraph/g_test.parquet',R+'data/test/transactions.parquet',q.TRANSACTION_HASH);Xte=Xte[Xtr.columns]