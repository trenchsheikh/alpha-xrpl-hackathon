import pandas as pd,numpy as np,sys
from lightgbm import LGBMClassifier
sys.path.insert(0,'.');from inj import inject
R='/workspace/dev-tester/alpha-xrpl-hackathon/'
l=pd.read_csv(R+'data/train/labels.csv');y=l.is_suspicious.values
q=pd.read_csv(R+'out/sub/q4_blend.csv')
def X(fb,fg,ids):
  B=pd.read_parquet(fb).set_index('TRANSACTION_HASH').loc[ids].select_dtypes(exclude='object')
  G=pd.read_parquet(fg).set_index('TRANSACTION_HASH').loc[ids].drop(columns=['ACCOUNT','dst'])
  return pd.concat([B,G],axis=1)
Xtr=X(R+'cache/feats_motif_train.parquet','g_train.parquet',l.TRANSACTION_HASH);Xte=X(R+'cache/test_feats_motif.parquet','g_test.parquet',q.TRANSACTION_HASH)[Xtr.columns]
P=dict(n_estimators=400,learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.5,verbose=-1,random_state=42,n_jobs=8)
m=LGBMClassifier(**P).fit(Xtr.reset_index(drop=True),y);p=m.predict_proba(Xte.reset_index(drop=True))[:,1]
qr=q.score.rank(pct=True).values;pr=pd.Series(p).rank(pct=True).values
outs={'h_graph-rankavg':.5*qr+.5*pr,'h_graph-inject40':inject(q.score.values,p,40,200)}
for n,s in outs.items():
  s=(s-s.min())/(s.max()-s.min());assert np.isfinite(s).all()
  pd.DataFrame({'TRANSACTION_HASH':q.TRANSACTION_HASH,'score':s}).to_csv(R+f'out/sub/{n}.csv',index=False)
  t=set(np.argsort(-s)[:401]);print(n,'top401 overlap with q4',len(t&set(np.argsort(-q.score.values)[:401])))
import zipfile
for n in outs:
  with zipfile.ZipFile(R+f'out/sub/code_{n}.zip','w') as z:
    for f in ['gfeat.py','linker.py','run.py','inj.py','make_sub.py']: z.write(f)
