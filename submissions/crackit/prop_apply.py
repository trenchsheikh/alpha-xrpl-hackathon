"""Cluster-max + link-neighbour propagation.
score' = x + 0.15*(nbmean_{p>0.7}(x) - x) where x = 0.4*rank(own) + 0.6*max rank in capped cluster (cap15,minp0.5)
usage: prop_apply.py train|test IN.csv(TRANSACTION_HASH,score) OUT.csv"""
import sys;sys.path.insert(0,'/workspace/crackit')
import numpy as np,pandas as pd
from scipy.stats import rankdata
from scipy.sparse import coo_matrix
from lk_greedy import greedy
w,inp,out=sys.argv[1:4];k='tr' if w=='train' else 'te'
t=pd.read_pickle(f'/tmp/lk_{k}.pkl');P=pd.read_pickle(f'/tmp/lk_{"P" if k=="tr" else "Pt"}.pkl');p=np.load(f'/tmp/lk_{"p" if k=="tr" else "pt"}.npy')
S=pd.read_csv(inp);n=len(S);r=rankdata(S.score)/n
ix=pd.Series(np.arange(n),index=S.TRANSACTION_HASH.values).reindex(t.TRANSACTION_HASH.values).values;ok=~np.isnan(ix)
ia=ix[P.ia.values];ib=ix[P.ib.values];m=~(np.isnan(ia)|np.isnan(ib))&(p>.7);ia=ia[m].astype(int);ib=ib[m].astype(int);pw=p[m]
c=greedy(len(t),P,p,15,.5);cc=np.full(n,-1);cc[ix[ok].astype(int)]=c[ok];cc[cc<0]=-np.arange(1,(cc<0).sum()+1)
cm=pd.Series(r).groupby(cc).transform('max').values;x=.4*r+.6*cm
W=coo_matrix((np.r_[pw,pw],(np.r_[ia,ib],np.r_[ib,ia])),shape=(n,n)).tocsr();deg=np.asarray(W.sum(1)).ravel()
nb=W@x/np.maximum(deg,1e-9);x=np.where(deg>0,.85*x+.15*nb,x)
pd.DataFrame({'TRANSACTION_HASH':S.TRANSACTION_HASH,'score':x}).to_csv(out,index=False)
