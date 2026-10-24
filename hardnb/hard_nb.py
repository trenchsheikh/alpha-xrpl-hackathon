# Hard-vs-rest model with neighbourhood-relative (scale-free) features; scenario/cohort-grouped CV; inject into base order.
import pandas as pd,numpy as np,glob,sys,zipfile
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
R='/workspace/dev-tester/alpha-xrpl-hackathon/';sys.path.insert(0,R+'lkgraph');from inj import inject
l=pd.read_csv(R+'data/train/labels.csv');y=l.is_suspicious.values;hard=((l.difficulty=='hard')&(y==1)).values.astype(int)
grp=l.cohort_id.fillna(l.scenario_id).fillna(pd.Series('bg'+(l.index//200).astype(str),index=l.index)).values
q=pd.read_csv(R+'out/sub/h_graph3s-inject40.csv')
def feats(fm,fg,ids):
  B=pd.read_parquet(fm).set_index('TRANSACTION_HASH').loc[ids]
  G=pd.read_parquet(fg).set_index('TRANSACTION_HASH').loc[ids]
  num=pd.concat([B.select_dtypes(exclude='object'),G.drop(columns=['ACCOUNT','dst'])],axis=1).reset_index(drop=True)
  acc=G.ACCOUNT.values;dst=G.dst.values;X=num.copy()
  key=['log_usd','relay_match6h','prior_in_delay','next_out_delay','prior_in_vs_next_out6h','relay_amount_matching','upstream_best_match_21600','downstream_best_match_21600','fundflow_continuity_4hop','s_lag_in','s_ratio_in','d_lag_out','d_ratio_out','hour','s_burst6','d_burst6']
  key=[k for k in key if k in num]
  for who,a in [('S',acc),('D',dst)]:
    gb=num[key].groupby(a)
    X[[f'{who}rk_{k}' for k in key]]=gb.rank(pct=True).values
    X[[f'{who}dz_{k}' for k in key]]=(num[key]-gb.transform('median')).values/(gb.transform('std').values+1e-9)
  # global pct ranks (scale-free across splits)
  X[[f'g_{k}' for k in key]]=num[key].rank(pct=True).values
  return X
Xtr=feats(R+'cache/feats_motif_train.parquet',R+'lkgraph/g_train.parquet',l.TRANSACTION_HASH)
Xte=feats(R+'cache/test_feats_motif.parquet',R+'lkgraph/g_test.parquet',q.TRANSACTION_HASH)[Xtr.columns]
P=dict(n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=20,subsample=.8,subsample_freq=1,colsample_bytree=.4,verbose=-1,n_jobs=8)
pte=LGBMClassifier(**P,random_state=42).fit(Xtr,hard).predict_proba(Xte)[:,1]
imp=pd.Series(LGBMClassifier(**P,random_state=42).fit(Xtr,hard).feature_importances_,Xtr.columns).nlargest(12);print(imp)
if True:
  for K in [20,40,80]:
    n=f'i_hardnb-inject{K}';s=inject(q.score.values,pte,K,200);s=(s-s.min())/(s.max()-s.min())
    assert np.isfinite(s).all() and len(s)==40101 and s.min()>=0 and s.max()<=1
    pd.DataFrame({'TRANSACTION_HASH':q.TRANSACTION_HASH,'score':s}).to_csv(R+f'out/sub/{n}.csv',index=False)
    with zipfile.ZipFile(R+f'out/sub/code_{n}.zip','w') as z:
      z.write(__file__,'hard_nb.py');z.write(R+'lkgraph/inj.py','inj.py');z.write(R+'lkgraph/gfeat.py','gfeat.py')
    print(n,'written')
