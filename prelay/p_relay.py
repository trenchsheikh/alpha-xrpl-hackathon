# Two-stage: P(relay family) x P(Hard | relay family); inject into h_graph3s-inject40 at rank 380.
import pandas as pd,numpy as np,sys,zipfile
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
R='/workspace/dev-tester/alpha-xrpl-hackathon/';sys.path.insert(0,R+'lkgraph');from inj import inject
l=pd.read_csv(R+'data/train/labels.csv')
def load(m,g):
  a=pd.read_parquet(R+m);b=pd.read_parquet(R+g).drop(columns=['ACCOUNT','dst'],errors='ignore')
  return a.merge(b,on='TRANSACTION_HASH',how='left',suffixes=('','_g'))
tr=l[['TRANSACTION_HASH']].merge(load('cache/feats_motif_train.parquet','lkgraph/g_train.parquet'),on='TRANSACTION_HASH',how='left')
te=load('cache/test_feats_motif.parquet','lkgraph/g_test.parquet')
cols=[c for c in tr.columns if c!='TRANSACTION_HASH' and c in te.columns and pd.api.types.is_numeric_dtype(tr[c])]
for d in (tr,te): d['amt_rank']=d['log_usd'].rank(pct=True)
cols+=['amt_rank'];X=tr[cols].astype(float).values;XT=te[cols].astype(float).values
fam=l.family.isin(['aggregate_relay','distributed_relay']).values.astype(int)
hard=((l.difficulty=='hard')&(l.is_suspicious==1)).values.astype(int)
grp=l.scenario_id.fillna(pd.Series('bg'+(l.index//200).astype(str),index=l.index)).values
P=dict(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1)
def fit(Xa,ya,ga,XTa):
  oof=np.zeros(len(ya));cv=StratifiedGroupKFold(5,shuffle=True,random_state=0)
  for a,b in cv.split(Xa,ya,ga): oof[b]=LGBMClassifier(**P).fit(Xa[a],ya[a]).predict_proba(Xa[b])[:,1]
  return oof,LGBMClassifier(**P).fit(Xa,ya).predict_proba(XTa)[:,1]
f_oof,f_te=fit(X,fam,grp,XT)
print('family top-2000 precision',fam[np.argsort(-f_oof)[:2000]].mean(),'pred test >0.5',(f_te>0.5).sum())
m=fam==1
h_oof=np.zeros(len(l));h_oof_r,h_te=fit(X[m],hard[m],grp[m],XT);h_oof[m]=h_oof_r
# for non-relay train rows use stage-2 model trained on relay rows (approx; low weight via f anyway)
h_oof[~m]=LGBMClassifier(**P).fit(X[m],hard[m]).predict_proba(X[~m])[:,1]
s_oof=f_oof*h_oof;s_te=f_te*h_te
print('train Hard P@40',hard[np.argsort(-s_oof)[:40]].mean(),'P@401',hard[np.argsort(-s_oof)[:401]].sum())
base=pd.read_csv(R+'out/sub/h_graph3s-inject40.csv');q=pd.read_csv(R+'out/sub/q4_blend.csv')
assert set(base.TRANSACTION_HASH)==set(q.TRANSACTION_HASH)
st=pd.Series(s_te,index=te.TRANSACTION_HASH).reindex(base.TRANSACTION_HASH).fillna(0).values
for K in [20,40]:
  s=inject(base.score.values,st,K,380);s=(s-s.min())/(s.max()-s.min())
  o=pd.DataFrame({'TRANSACTION_HASH':base.TRANSACTION_HASH,'score':s});assert len(o)==40101 and np.isfinite(s).all()
  o.to_csv(R+f'out/sub/p_relay{K}.csv',index=False)
  with zipfile.ZipFile(R+f'out/sub/code_p_relay{K}.zip','w') as z: z.write(R+'prelay/p_relay.py','p_relay.py');z.write(R+'lkgraph/inj.py','inj.py')
  print('wrote',K)
