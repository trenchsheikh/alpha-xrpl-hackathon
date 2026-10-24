import pandas as pd, numpy as np, lightgbm as lgb, zipfile, os
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
S='out/sub/'; q=pd.read_csv(S+'q4_blend.csv')
X=pd.read_parquet('cache/test_feats_motif.parquet').set_index('TRANSACTION_HASH').reindex(q.TRANSACTION_HASH)
X=X.select_dtypes(include=[np.number]).astype('float32')
r=rankdata(-q.score.values,method='ordinal')  # 1=top
rq=rankdata(q.score.values)/len(q)
for N,alist in [(300,[0.3,0.5]),(500,[0.3]),(200,[0.3])]:
    lab=np.full(len(q),-1); lab[r<=N]=1; lab[r>4000]=0
    pred=np.zeros(len(q)); cnt=np.zeros(len(q))
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    strat=np.where(lab==-1,2,lab)
    for tr,va in skf.split(X,strat):
        t=tr[lab[tr]>=0]
        m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,verbose=-1)
        m.fit(X.values[t],lab[t]); pred[va]=m.predict_proba(X.values[va])[:,1]
    rp=rankdata(pred)/len(q)
    for a in alist:
        f=(1-a)*rq+a*rp; f=(f-f.min())/(f.max()-f.min())
        name=f'pl_{N}_{str(a).replace(".","")}'
        pd.DataFrame({'TRANSACTION_HASH':q.TRANSACTION_HASH,'score':f}).to_csv(S+name+'.csv',index=False)
        with zipfile.ZipFile(S+'code_'+name+'.zip','w') as z: z.write(__file__,'pl_selftrain.py')
        top=lambda s:set(np.argsort(-s)[:401])
        print(name,'overlap401',len(top(f)&top(q.score.values)),'pseudo-only overlap',len(top(pred)&top(q.score.values)),flush=True)
