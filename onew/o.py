import sys;exec(open('/workspace/dev-tester/alpha-xrpl-hackathon/onew/feat_part.py').read())
from sklearn.metrics import average_precision_score as ap
mtr=pd.read_parquet(R+'cache/feats_d510ebca5bed.parquet').set_index('TRANSACTION_HASH').loc[l.TRANSACTION_HASH].select_dtypes(exclude='object')
mte=pd.read_parquet(R+'cache/test_feats_medium_windows.parquet').set_index('TRANSACTION_HASH').loc[q.TRANSACTION_HASH][mtr.columns]
new=[c for c in mtr.columns if c not in Xtr.columns]
Xtr=pd.concat([Xtr,mtr[new].reset_index(drop=True).add_prefix('mw_')],axis=1);Xte=pd.concat([Xte,mte[new].reset_index(drop=True).add_prefix('mw_')],axis=1)
Xtr.columns=Xte.columns=[f'f{i}' for i in range(Xtr.shape[1])]
print('nfeat',Xtr.shape,flush=True)
fold=pd.read_csv(R+'out/cv/repo-main_oof.csv').fold.values
P=dict(n_estimators=3000,learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.4,verbose=-1,n_jobs=8)
import lightgbm as lgb
oof=np.zeros(len(y));pte=np.zeros(len(Xte));its=[]
for sd in range(5):
  o=np.zeros(len(y))
  for k in range(5):
    tr=fold!=k
    m=LGBMClassifier(**P,random_state=sd).fit(Xtr[tr],y[tr],eval_set=[(Xtr[~tr],y[~tr])],eval_metric='average_precision',callbacks=[lgb.early_stopping(150,verbose=False)])
    o[~tr]=m.predict_proba(Xtr[~tr])[:,1];its.append(m.best_iteration_)
  print('seed',sd,'AP',round(ap(y,o),4),flush=True);oof+=pd.Series(o).rank(pct=True).values
print('CV AP 5-seed',round(ap(y,oof),4),'iters',its,flush=True)
top=np.argsort(-oof)[:401];print({d:round(((l.difficulty==d)&(y==1)).values[top].sum()/((l.difficulty==d)&(y==1)).sum(),3) for d in['easy','medium','hard']})
P['n_estimators']=int(np.median(its)*1.1)
for sd in range(5): pte+=pd.Series(LGBMClassifier(**P,random_state=sd).fit(Xtr,y).predict_proba(Xte)[:,1]).rank(pct=True).values
np.save(R+'onew/pte.npy',pte);np.save(R+'onew/oof.npy',oof)
h=pd.read_csv(R+'out/sub/h_graph3s-inject40.csv');assert (h.TRANSACTION_HASH.values==q.TRANSACTION_HASH.values).all()
q4=pd.read_csv(R+'out/sub/q4_blend.csv');assert set(q4.TRANSACTION_HASH)==set(h.TRANSACTION_HASH)
hs=h.score.values;N=len(hs)
def sc(order):
  s=np.empty(N);s[order]=1-np.arange(N)/N;return s
full=sc(np.argsort(-pte,kind='stable'))
ho=list(np.argsort(-hs,kind='stable'));t=set(ho[:401])
tail=sc(ho[:401]+[i for i in np.argsort(-pte,kind='stable') if i not in t])
inj=inject(hs,pte,40,200)
for n,s in [('o_full',full),('o_tail',tail),('o_inj40',inj)]:
  s=(s-s.min())/(s.max()-s.min());assert np.isfinite(s).all()
  pd.DataFrame({'TRANSACTION_HASH':h.TRANSACTION_HASH,'score':s}).to_csv(R+f'out/sub/{n}.csv',index=False)
  with zipfile.ZipFile(R+f'out/sub/code_{n}.zip','w') as z: z.write(R+'onew/o.py','o.py');z.write(R+'onew/feat_part.py','feat_part.py');z.write(R+'lkgraph/inj.py','inj.py')
  print('wrote',n,flush=True)
