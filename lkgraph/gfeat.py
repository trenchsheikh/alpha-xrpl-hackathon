import pandas as pd,numpy as np,json,sys
sys.path.insert(0,'.');from linker import load
D='/workspace/dev-tester/alpha-xrpl-hackathon/data/'
def gf(t):
  t=t.copy();A,B=t.ACCOUNT,t.dst
  outd=A.value_counts();ind=B.value_counts()
  uo=t.groupby('ACCOUNT').dst.nunique();ui=t.groupby('dst').ACCOUNT.nunique()
  f=pd.DataFrame(index=t.index)
  for n,s in [('s_out',outd),('s_in',ind),('s_uout',uo),('s_uin',ui)]: f[n]=A.map(s).fillna(0)
  for n,s in [('d_out',outd),('d_in',ind),('d_uout',uo),('d_uin',ui)]: f[n]=B.map(s).fillna(0)
  pr=t.groupby(['ACCOUNT','dst']).size()
  f['pair_n']=pr.reindex(list(zip(A,B))).values
  f['recip']=pr.reindex(list(zip(B,A))).fillna(0).values
  # relay: dst forwards later; sender received earlier
  t=t.sort_values('ts')
  for who,col in [('s','ACCOUNT'),('d','dst')]:
    g=t.groupby(col).ts
    f.loc[t.index,who+'_gap_prev']=g.diff().values;f.loc[t.index,who+'_gap_next']=(-g.diff(-1)).values
  # time from sender's last incoming to this send; dst's next outgoing after this
  inc=t[['dst','ts','u']].rename(columns={'dst':'acc'});out=t[['ACCOUNT','ts','u']].rename(columns={'ACCOUNT':'acc'})
  x=pd.merge_asof(t[['ACCOUNT','ts']].reset_index().rename(columns={'ACCOUNT':'acc'}).sort_values('ts'),inc.rename(columns={'ts':'tin','u':'uin'}).sort_values('tin'),left_on='ts',right_on='tin',by='acc',allow_exact_matches=False).set_index('index')
  f['s_lag_in']=t.ts-x.tin;f['s_ratio_in']=np.log((t.u+1)/(x.uin+1))
  y=pd.merge_asof(t[['dst','ts']].reset_index().rename(columns={'dst':'acc'}).sort_values('ts'),out.rename(columns={'ts':'tout','u':'uout'}).sort_values('tout'),left_on='ts',right_on='tout',by='acc',direction='forward',allow_exact_matches=False).set_index('index')
  f['d_lag_out']=y.tout-t.ts;f['d_ratio_out']=np.log((y.uout+1)/(t.u+1))
  # bursts
  for col,p in [('ACCOUNT','s'),('dst','d')]:
    for w in [1,6,24]:
      c=np.zeros(len(t))
      for _,g in t.groupby(col):
        v=g.ts.values;c[t.index.get_indexer(g.index)]=np.searchsorted(v,v+w)-np.searchsorted(v,v-w)
      f[f'{p}_burst{w}']=pd.Series(c,index=t.index)
  f['lu']=np.log1p(t.u)
  # 2-hop cycle: B->C->A exists
  succ=t.groupby('ACCOUNT').dst.agg(set).to_dict()
  f['cyc3']=[int(any(A_ in succ.get(c,()) for c in succ.get(B_,()))) for A_,B_ in zip(A,B)]
  return f
def hop(t,seed,maxh=4):
  import collections
  adj=collections.defaultdict(set)
  for a,b in zip(t.ACCOUNT,t.dst): adj[a].add(b);adj[b].add(a)
  d={s:0 for s in seed};fr=list(seed)
  for h in range(1,maxh+1):
    nf=[]
    for u in fr:
      for v in adj[u]:
        if v not in d: d[v]=h;nf.append(v)
    fr=nf
  return t.ACCOUNT.map(d).fillna(9).values,t.dst.map(d).fillna(9).values
if __name__=='__main__':
  for w in ['train','test']:
    t=load(D+w);print(w,len(t),flush=True);f=gf(t);f['TRANSACTION_HASH']=t.TRANSACTION_HASH.values;f['ACCOUNT']=t.ACCOUNT.values;f['dst']=t.dst.values
    f.to_parquet(f'g_{w}.parquet')
