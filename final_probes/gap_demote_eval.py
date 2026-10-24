import pandas as pd,numpy as np,json
from sklearn.metrics import average_precision_score as AP
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
def groups(path):
    d=pd.read_parquet(path,columns=['TRANSACTION_HASH','ACCOUNT','TIMESTAMP','EXTRA_TRANSACTION_FIELDS'])
    d['rcv']=d.EXTRA_TRANSACTION_FIELDS.map(lambda s: (json.loads(s) or {}).get('Destination') if isinstance(s,str) else None)
    d['t']=pd.to_numeric(d.TIMESTAMP).astype(float); d=d.reset_index(drop=True); d['i']=d.index
    a=d[d.rcv.notna()][['i','rcv','t']]; b=d[['i','ACCOUNT','t']]
    m=a.merge(b,left_on='rcv',right_on='ACCOUNT',suffixes=('a','b'))
    m['g']=m.tb-m.ta; m=m[(m.g>0)&(m.g<=3600)]
    n=len(d); A=coo_matrix((np.ones(len(m)),(m.ia,m.ib)),shape=(n,n))
    _,lab=connected_components(A,directed=False); d['grp']=lab
    m['grp']=lab[m.ia.values]; med=m.groupby('grp').g.median()
    d['med']=d.grp.map(med); d['gs']=d.grp.map(d.grp.value_counts())
    return d[['TRANSACTION_HASH','grp','med','gs']]
def apply(df,f=1,w=0):
    s=df.score.copy(); band=df.med.between(100,400)
    s[band]*=f
    if w:
        gm=s.groupby(df.grp).transform('max'); big=df.gs>=2
        s[big]=(1-w)*s[big]+w*gm[big]
    return s,band
lab=pd.read_csv('xrpl-data/train/labels.csv'); print(lab.columns.tolist())
oof=pd.read_csv('dev-tester/alpha-xrpl-hackathon/out/cv/repo-agg-wide_oof.csv')
y=[c for c in lab.columns if c not in('TRANSACTION_HASH','cohort_id')][0]
tr=oof.merge(lab[['TRANSACTION_HASH',y]],on='TRANSACTION_HASH').merge(groups('xrpl-data/train/transactions.parquet'),on='TRANSACTION_HASH',how='left')
print('label col',y,len(oof),len(tr)); yy=tr[y].astype(int)
base=AP(yy,tr.score); print('baseline',round(base,5))
band=tr.med.between(100,400); print('train band rows',band.sum(),'fraud in band',yy[band].sum(),'of',yy.sum())
win=[]
for f in(.5,.8,.9):
    a=AP(yy,apply(tr,f=f)[0]); print('demote',f,round(a,5)); win+= [('n1_gapdemote%s'%f,f,0)] if a>base else []
for w in(.1,.2,.3):
    a=AP(yy,apply(tr,w=w)[0]); print('groupmax',w,round(a,5)); win+= [('n2_groupmax%s'%w,1,w)] if a>base else []
sub=pd.read_csv('dev-tester/alpha-xrpl-hackathon/out/sub/k2_blend.csv')
te=sub.merge(groups('xrpl-data/test/transactions.parquet'),on='TRANSACTION_HASH',how='left')
b=te.med.between(100,400); print('test band groups',te[b].grp.nunique(),'rows',b.sum())
for name,f,w in win:
    o=sub.copy(); o['score']=apply(te,f,w)[0].values
    assert len(o)==len(sub) and (o.TRANSACTION_HASH==sub.TRANSACTION_HASH).all() and o.score.notna().all()
    p='dev-tester/alpha-xrpl-hackathon/out/sub/%s.csv'%name; o.to_csv(p,index=False); print('wrote',p)
