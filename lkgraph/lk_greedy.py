import pandas as pd,numpy as np,sys;sys.path.insert(0,'/workspace/crackit')
from sklearn.metrics import roc_auc_score as auc,average_precision_score as ap
def greedy(n,P,p,cap,minp):
  par=np.arange(n);sz=np.ones(n,int)
  def f(x):
    while par[x]!=x: par[x]=par[par[x]];x=par[x]
    return x
  o=np.argsort(-p);ia=P.ia.values;ib=P.ib.values
  for e in o:
    if p[e]<minp: break
    a=f(ia[e]);b=f(ib[e])
    if a!=b and sz[a]+sz[b]<=cap:
      if sz[a]<sz[b]: a,b=b,a
      par[b]=a;sz[a]+=sz[b]
  return np.array([f(x) for x in range(n)])
def cov(t,c):
  t=t.assign(c=c).sort_values('ts');sz=t.c.value_counts();res={}
  for cc,g in t[t.c.map(sz)>=6].groupby('c'):
    mids=set(g.ACCOUNT)&set(g.dst)
    if not mids: continue
    bal={};unc=tot=0
    for a,b,u in zip(g.ACCOUNT,g.dst,g.u):
      if a in mids: tot+=1;unc+=bal.get(a,0)<u*.999
      bal[b]=bal.get(b,0)+u;bal[a]=bal.get(a,0)-u
    res[cc]=1-unc/tot
  return t.c.map(res).fillna(-1).sort_index().values
if __name__=='__main__':
  tr=pd.read_pickle('/tmp/lk_tr.pkl');P=pd.read_pickle('/tmp/lk_P.pkl');p=np.load('/tmp/lk_p.npy')
  for cap,minp in [(17,.3),(20,.3),(15,.5),(20,.1)]:
    c=greedy(len(tr),P,p,cap,minp);s=cov(tr,c);d=tr.assign(s=s,c=c)
    r=d[d.family.isin(['aggregate_relay','distributed_relay'])]
    top=r.groupby('scenario_id').c.agg(lambda x:x.value_counts().index[0])
    pur=np.median([(d[d.c==cc].scenario_id==sid).mean() for sid,cc in top.items()])
    rec=r.groupby('scenario_id').apply(lambda g:g.c.value_counts().iloc[0]/len(g)).median()
    h=d[((d.difficulty=='hard')&(d.family!='ambiguous'))|(d.is_suspicious==0)]
    ra=r[r.family=='aggregate_relay'];rd=r[r.family=='distributed_relay']
    print(cap,minp,'pur',round(pur,2),'compl',round(rec,2),'aggAUC',round(auc(ra.is_suspicious,ra.s),3),'distAUC',round(auc(rd.is_suspicious,rd.s),3),'hardAP',round(ap(h.is_suspicious,h.s),4),'AUC',round(auc(h.is_suspicious,h.s),3),flush=True)
    d[['TRANSACTION_HASH','s']].to_csv(f'/tmp/g_{cap}_{minp}.csv',index=False)
