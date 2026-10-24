# small-K / rank-300 graph injections (seed 42) + report for all h_graph variants
import pandas as pd,numpy as np,glob,sys,zipfile
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0,'.');from inj import inject
R='/workspace/dev-tester/alpha-xrpl-hackathon/'
l=pd.read_csv(R+'data/train/labels.csv');y=l.is_suspicious.values
q=pd.read_csv(R+'out/sub/q4_blend.csv')
def X(fb,fg,ids):
  B=pd.read_parquet(fb).set_index('TRANSACTION_HASH').loc[ids].select_dtypes(exclude='object')
  G=pd.read_parquet(fg).set_index('TRANSACTION_HASH').loc[ids].drop(columns=['ACCOUNT','dst'])
  return pd.concat([B,G],axis=1)
Xtr=X(R+'cache/feats_motif_train.parquet','g_train.parquet',l.TRANSACTION_HASH).reset_index(drop=True)
Xte=X(R+'cache/test_feats_motif.parquet','g_test.parquet',q.TRANSACTION_HASH)[Xtr.columns].reset_index(drop=True)
P=dict(n_estimators=400,learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.5,verbose=-1,random_state=42,n_jobs=8)
p=LGBMClassifier(**P).fit(Xtr,y).predict_proba(Xte)[:,1]
base=sum(pd.read_csv(f).score.rank(pct=True).values for f in sorted(glob.glob(R+'out/cv/repo-*_oof.csv')))
o=pd.read_parquet('oof.parquet')['base+g'].values
q4top=set(np.argsort(-q.score.values)[:401])
print('proxy q4-proxy AP',round(ap(y,base),4),flush=True)
for K,pos in [(20,200),(30,200),(40,300),(60,300),(20,300)]:
  n=f'h_graph-inject{K}'+('' if pos==200 else f'r{pos}')
  s=inject(q.score.values,p,K,pos);s=(s-s.min())/(s.max()-s.min());assert np.isfinite(s).all()
  pd.DataFrame({'TRANSACTION_HASH':q.TRANSACTION_HASH,'score':s}).to_csv(R+f'out/sub/{n}.csv',index=False)
  with zipfile.ZipFile(R+f'out/sub/code_{n}.zip','w') as z:
    for f in ['gfeat.py','linker.py','run.py','inj.py','make_sub.py','make_sub2.py','make_sub3.py']: z.write(f)
for K,pos,n in [(20,200,'inject20'),(30,200,'inject30'),(40,200,'inject40'),(40,300,'inject40r300'),(20,300,'inject20r300'),(60,300,'inject60r300'),(60,200,'inject60'),(80,200,'inject80'),(120,200,'inject120'),(40,100,'inject40r100')]:
  d=pd.read_csv(R+f'out/sub/h_graph-{n}.csv');s=d.score.values
  assert (d.TRANSACTION_HASH==q.TRANSACTION_HASH).all() and np.isfinite(s).all() and s.min()>=0 and s.max()<=1
  print(n,'overlap',len(set(np.argsort(-s)[:401])&q4top),'proxyAP',round(ap(y,inject(base,o,K,pos)),4),flush=True)
d=pd.read_csv(R+'out/sub/h_graph3s-inject40.csv');s=d.score.values
assert (d.TRANSACTION_HASH==q.TRANSACTION_HASH).all() and np.isfinite(s).all() and s.min()>=0 and s.max()<=1
print('3s-inject40 overlap',len(set(np.argsort(-s)[:401])&q4top))
