# variants of graph inject; train-proxy AP = inject applied to OOF (repo-* rank-sum as q4 proxy, base+g OOF as graph model)
import pandas as pd,numpy as np,glob,sys,zipfile
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0,'.');from inj import inject
R='/workspace/dev-tester/alpha-xrpl-hackathon/'
l=pd.read_csv(R+'data/train/labels.csv');y=l.is_suspicious.values
fold=pd.read_csv(R+'out/cv/repo-main_oof.csv').fold.values
q=pd.read_csv(R+'out/sub/q4_blend.csv')
def X(fb,fg,ids):
  B=pd.read_parquet(fb).set_index('TRANSACTION_HASH').loc[ids].select_dtypes(exclude='object')
  G=pd.read_parquet(fg).set_index('TRANSACTION_HASH').loc[ids].drop(columns=['ACCOUNT','dst'])
  return pd.concat([B,G],axis=1)
Xtr=X(R+'cache/feats_motif_train.parquet','g_train.parquet',l.TRANSACTION_HASH).reset_index(drop=True)
Xte=X(R+'cache/test_feats_motif.parquet','g_test.parquet',q.TRANSACTION_HASH)[Xtr.columns].reset_index(drop=True)
P=dict(n_estimators=400,learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.5,verbose=-1,n_jobs=8)
base=sum(pd.read_csv(f).score.rank(pct=True).values for f in sorted(glob.glob(R+'out/cv/repo-*_oof.csv')))
p={};o={}
for sd in [42,7,123]:
  p[sd]=LGBMClassifier(**P,random_state=sd).fit(Xtr,y).predict_proba(Xte)[:,1]
  oo=np.zeros(len(y))
  for k in range(5):
    tr=fold!=k;oo[~tr]=LGBMClassifier(**P,random_state=sd).fit(Xtr[tr],y[tr]).predict_proba(Xtr[~tr])[:,1]
  o[sd]=oo
p3=np.mean([pd.Series(v).rank(pct=True).values for v in p.values()],0)
o3=np.mean([pd.Series(v).rank(pct=True).values for v in o.values()],0)
print('proxy base AP',round(ap(y,base),4),'| inject40@200 (seed42)',round(ap(y,inject(base,o[42],40,200)),4))
V={'h_graph-inject20':(p[42],o[42],20,200),'h_graph-inject30':(p[42],o[42],30,200),'h_graph-inject40r300':(p[42],o[42],40,300),'h_graph-inject60r300':(p[42],o[42],60,300),'h_graph-inject60':(p[42],o[42],60,200),'h_graph-inject80':(p[42],o[42],80,200),'h_graph-inject120':(p[42],o[42],120,200),
   'h_graph-inject40r100':(p[42],o[42],40,100),'h_graph3s-inject40':(p3,o3,40,200)}
q4top=set(np.argsort(-q.score.values)[:401])
for n,(pt,ot,K,pos) in V.items():
  s=inject(q.score.values,pt,K,pos);s=(s-s.min())/(s.max()-s.min());assert np.isfinite(s).all() and len(s)==len(q)
  d=pd.DataFrame({'TRANSACTION_HASH':q.TRANSACTION_HASH,'score':s});d.to_csv(R+f'out/sub/{n}.csv',index=False)
  assert (pd.read_csv(R+f'out/sub/{n}.csv').TRANSACTION_HASH==q.TRANSACTION_HASH).all()
  print(n,'overlap',len(set(np.argsort(-s)[:401])&q4top),'proxyAP',round(ap(y,inject(base,ot,K,pos)),4))
  with zipfile.ZipFile(R+f'out/sub/code_{n}.zip','w') as z:
    for f in ['gfeat.py','linker.py','run.py','inj.py','make_sub.py','make_sub2.py']: z.write(f)
