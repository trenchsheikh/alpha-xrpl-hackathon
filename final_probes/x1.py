import pandas as pd,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score as ap
tr=pd.read_pickle('tr.pkl');te=pd.read_pickle('te.pkl')
F=['cx','lu','round','s_out','s_in','r_out','r_in','s_age','r_age','s_new','s_cv','r_cv','r_nsend','s_ndst','cyc','s_ncur']
S='dev-tester/alpha-xrpl-hackathon/out/sub/'
L=pd.read_csv('xrpl-data/train/labels.csv')[['TRANSACTION_HASH','difficulty']]
tr=tr.drop(columns=[c for c in ['difficulty'] if c in tr]).merge(L,on='TRANSACTION_HASH',how='left')
y=tr.is_suspicious if 'is_suspicious' in tr else tr.merge(pd.read_csv('xrpl-data/train/labels.csv')[['TRANSACTION_HASH','is_suspicious']],on='TRANSACTION_HASH').is_suspicious
T={'easy':tr.difficulty.eq('easy')&(y==1),'med':tr.difficulty.eq('medium')&(y==1),'hard':tr.difficulty.isin(['hard','ambiguous'])&(y==1)}
base=pd.read_csv(S+'n1_blend.csv')
rb=base.score.rank(pct=True).values
P={}
def chk(o,n):
    assert len(o)==40101 and o.TRANSACTION_HASH.is_unique and not o.isna().any().any();o.to_csv(S+n+'.csv',index=False);print('wrote',n,flush=True)
for k in ['med','hard','easy']:
    m=H(max_iter=200,learning_rate=.05,random_state=0).fit(tr[F],T[k].astype(int))
    P[k]=pd.Series(m.predict_proba(te[F])[:,1],index=te.TRANSACTION_HASH).reindex(base.TRANSACTION_HASH).fillna(0).rank(pct=True).values
    if k=='hard':
        chk(pd.DataFrame({'TRANSACTION_HASH':base.TRANSACTION_HASH,'score':np.maximum.reduce([rb,.8*P['med'],.8*P['hard']])}),'x1_experts')
        chk(pd.DataFrame({'TRANSACTION_HASH':base.TRANSACTION_HASH,'score':np.maximum.reduce([rb,.9*P['med'],.7*P['hard']])}),'x2_experts')
# x3 quota interleave
hs=base.TRANSACTION_HASH.values
ob=list(hs[np.argsort(-rb)]);om=list(hs[np.argsort(-P['med'])]);oh=list(hs[np.argsort(-P['hard'])])
q={'b':int(1000*.66),'m':180,'h':160};picked=[];ps=set()
its={'b':iter(ob),'m':iter(om),'h':iter(oh)};cnt={'b':0,'m':0,'h':0}
while len(picked)<1000:
    for c in 'bmh':
        if cnt[c]<q[c]*(len(picked)+1)/1000+1:
            h=next(its[c])
            while h in ps: h=next(its[c])
            picked.append(h);ps.add(h);cnt[c]+=1
picked=picked[:1000];order=picked+[h for h in ob if h not in set(picked)]
chk(pd.DataFrame({'TRANSACTION_HASH':order,'score':np.linspace(1,0,len(order))}),'x3_experts')
# OOF 3-fold
o=pd.read_csv('dev-tester/alpha-xrpl-hackathon/out/cv/repo-agg-wide_oof.csv')[['TRANSACTION_HASH','score']]
R={}
for k in T: R[k]=pd.Series(cross_val_predict(H(max_iter=200,learning_rate=.05,random_state=0),tr[F],T[k].astype(int),cv=3,method='predict_proba')[:,1],index=tr.TRANSACTION_HASH)
d=o.set_index('TRANSACTION_HASH');d['y']=y.values if False else pd.Series(y.values,index=tr.TRANSACTION_HASH)
for k in R: d[k]=R[k].rank(pct=True)
d=d.dropna();a=d.score.rank(pct=True)
print('n',len(d),'a base',ap(d.y,d.score),'b maxexp',ap(d.y,d[['easy','med','hard']].max(1)),'c',ap(d.y,np.maximum.reduce([a,.8*d.med,.8*d.hard])),'x2',ap(d.y,np.maximum.reduce([a,.9*d.med,.7*d.hard])))
