import pandas as pd,numpy as np,glob,sys
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score as ap
from gfeat import hop
R='/workspace/dev-tester/alpha-xrpl-hackathon/'
l=pd.read_csv(R+'data/train/labels.csv');y=l.is_suspicious.values
fold=pd.read_csv(R+'out/cv/repo-main_oof.csv').fold.values
base=sum(pd.read_csv(f).score.rank(pct=True).values for f in sorted(glob.glob(R+'out/cv/repo-*_oof.csv')))
Btr=pd.read_parquet(R+'cache/feats_motif_train.parquet').set_index('TRANSACTION_HASH').loc[l.TRANSACTION_HASH]
Btr=Btr.select_dtypes(exclude='object')
G=pd.read_parquet('g_train.parquet').set_index('TRANSACTION_HASH').loc[l.TRANSACTION_HASH]
gt=pd.read_parquet('g_test.parquet');print('acct overlap',len((set(gt.ACCOUNT)|set(gt.dst)) & (set(G.ACCOUNT)|set(G.dst))))
gc=[c for c in G.columns if c not in('ACCOUNT','dst')]
def rec(s):
  top=np.argsort(-s)[:401];return {d:round(((l.difficulty==d)&(y==1)).values[top].sum()/((l.difficulty==d)&(y==1)).sum(),3) for d in['easy','medium','hard']}
def rep(n,s):print(n,'AP',round(ap(y,s),4),rec(np.asarray(s)),flush=True)
rep('baseline',base)
P=dict(n_estimators=400,learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.5,verbose=-1,random_state=42,n_jobs=8)
def cv(X,hopf=False):
  o=np.zeros(len(y))
  for k in range(5):
    tr=fold!=k;X2=X.copy()
    if hopf:
      seed=set(G.ACCOUNT[tr&(y==1)])|set(G.dst[tr&(y==1)])
      X2['hs'],X2['hd']=hop(G.assign(ts=0),seed)
    m=LGBMClassifier(**P).fit(X2[tr],y[tr]);o[~tr]=m.predict_proba(X2[~tr])[:,1]
  return o
res={}
for n,X,h in [('base',Btr,False),('base+g',pd.concat([Btr,G[gc]],axis=1),False)]:
  o=cv(X.reset_index(drop=True),h);res[n]=o;rep(n,o)
  for w in [.3,.5]: rep(f'  blend{w} w/ baseline',(1-w)*pd.Series(base).rank(pct=True)+w*pd.Series(o).rank(pct=True))
pd.DataFrame(res).assign(TRANSACTION_HASH=l.TRANSACTION_HASH.values).to_parquet('oof.parquet')
amb=l.family.values!='ambiguous'
print('baseline AP excl ambiguous',ap(y[amb],base[amb]))
