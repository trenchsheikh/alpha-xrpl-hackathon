import pandas as pd, numpy as np, zipfile
S='out/sub/'; C='/workspace/crackit/'
def top(p,k=401):
    d=pd.read_csv(p); return d.sort_values('score',ascending=False).TRANSACTION_HASH.iloc[:k].tolist()
base=pd.read_csv(S+'h_graph3s-inject40.csv').sort_values('score',ascending=False)
order=base.TRANSACTION_HASH.tolist(); q4=set(pd.read_csv(S+'q4_blend.csv').TRANSACTION_HASH)
assert set(order)==q4 and len(order)==40101
hi=[S+f for f in ['h_graph-inject40.csv','h_graph3s-inject40.csv','k_union.csv','h_graph-inject40r300.csv','h_graph-inject40r100.csv']]
hits=[set(top(p)) for p in hi]; cons=set.intersection(*hits)
rank={h:i for i,h in enumerate(order)}
low=[S+'r_rand20.csv',S+'i_hardnb-inject20.csv',S+'j_cohort-inject20.csv',C+'big_m18b40.csv',C+'nov_inj40.csv']+[S+f'pl_{x}.csv' for x in ['300_05','300_03','500_03','200_03']]
goodtop=set().union(*hits)|set(top(S+'q4_blend.csv'))
dem=set()
for p in low: dem|=set(top(p))-goodtop
dem-=cons
R=[{x:i for i,x in enumerate(top(p))} for p in hi]
first=sorted(cons,key=lambda h:np.mean([r[h] for r in R]))
rest=[h for h in order if h not in cons and h not in dem]
mid=rest[:401-len(first)]; final=first+mid+[h for h in order if h in dem]+rest[401-len(first):]
assert len(final)==40101 and len(set(final))==40101
pd.DataFrame({'TRANSACTION_HASH':final,'score':np.linspace(1,0,40101)}).to_csv(S+'f_lb1.csv',index=False)
moved=sum(1 for h in dem if rank[h]<401)
print('consensus',len(cons),'demoted_set',len(dem),'demoted from base top401',moved)
