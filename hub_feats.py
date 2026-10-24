"""Slow / hub-pruned scenario-shape features. Label-free, identical for every split, vectorised."""
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def _sparse(a, op):
    T=[a]; k=1
    while 2*k<=len(a): p=T[-1]; T.append(op(p[:-k], p[k:])); k*=2
    return T
def _query(T, lo, hi, op, empty):
    L=hi-lo; out=np.full(len(lo), empty, dtype=float); m=L>0
    j=np.zeros(len(lo),int); j[m]=np.floor(np.log2(L[m])).astype(int)
    for lv in np.unique(j[m]):
        s=m&(j==lv); t=T[lv]; out[s]=op(t[lo[s]], t[hi[s]-(1<<lv)])
    return out

class Side:
    """txs sorted by (key wallet, time); windows are contiguous index ranges."""
    def __init__(s, key, t):
        s.o=np.lexsort((t,key)); s.k=key[s.o]; s.t=t[s.o].astype(float)
        s.pos=np.empty(len(key),int); s.pos[s.o]=np.arange(len(key))
        s.kt=s.k*1e9+s.t  # combined monotone key (t < 1e9 relative)
    def rng(s, key_of, t0, t1):  # range of entries with k==key_of and t in [t0,t1]
        return np.searchsorted(s.kt, key_of*1e9+t0,'left'), np.searchsorted(s.kt, key_of*1e9+t1,'right')

def _gapcv(S, lo, hi):
    g=np.diff(S.t); g[S.k[1:]!=S.k[:-1]]=0
    c1=np.concatenate([[0],np.cumsum(g)]); c2=np.concatenate([[0],np.cumsum(g*g)])
    a=lo; b=np.maximum(hi-1,lo); n=b-a
    m1=np.where(n>0,(c1[b]-c1[a])/np.maximum(n,1),0); m2=np.where(n>0,(c2[b]-c2[a])/np.maximum(n,1),0)
    sd=np.sqrt(np.maximum(m2-m1*m1,0)); return np.where(n>=2, sd/(m1+1), -1.0), m1

def hub_features(d, aggW=1800):
    t0=d.TIMESTAMP.to_numpy(np.int64); t=(t0-t0.min()).astype(float); n=len(d)
    codes,uni=pd.factorize(pd.concat([d.FROM_ADDRESS.astype(str),d.TO_ADDRESS.astype(str)]))
    src=codes[:n].astype(np.int64); dst=codes[n:].astype(np.int64); W_=len(uni)
    asset=pd.factorize(d['ASSET'])[0].astype(float) if 'ASSET' in d else np.zeros(n)
    deg=np.bincount(src,minlength=W_)+np.bincount(dst,minlength=W_)
    out={}
    S=Side(src,t); R=Side(dst,t)
    pair=src*W_+dst; first=np.zeros(n); fo=np.lexsort((t,pair)); fp=pair[fo]
    first[fo[np.r_[True, fp[1:]!=fp[:-1]]]]=1
    for side,X,key,other in (('out',S,src,dst),('in',R,dst,src)):
        fs=np.concatenate([[0],np.cumsum(first[X.o])])
        oth=other[X.o]
        # distinct counterparties within window, via first-occurrence of (key,other) inside the key's sorted list
        for W in (1800,3600,7200,14400):
            lo,hi=X.rng(key,t-W,t+W)
            # distinct: count entries j in [lo,hi) whose previous same-counterparty entry is < lo
            pk=X.k*W_+oth; o2=np.lexsort((np.arange(n),pk)); prev=np.full(n,-1)
            same=np.r_[False,pk[o2][1:]==pk[o2][:-1]]; prev[o2[same]]=o2[np.r_[same[1:],False]] if False else 0
            # compute prev index properly
            prev=np.full(n,-1); idx=o2; s_=np.r_[False,pk[idx][1:]==pk[idx][:-1]]; prev[idx[s_]]=idx[np.r_[s_[1:],False]]
            # distinct = #{j in [lo,hi): prev[j]<lo}; approximate exactly for small ranges via sort-trick on blocks
            cnt=hi-lo; dist=cnt.astype(float)
            small=cnt<=64
            # loop over offsets (bounded 64) vectorised
            acc=np.zeros(n)
            for off in range(64):
                j=lo+off; v=small&(j<hi)
                jj=np.minimum(j,n-1); acc+=v&(prev[jj]<lo)
            dist[small]=acc[small]
            cv,mg=_gapcv(X,lo,hi)
            out[f'slow_f{side}_{W}']=dist; out[f'slow_f{side}_cv_{W}']=cv
            if W==7200:
                out[f'burst_{side}_new']=fs[hi]-fs[lo]; out[f'burst_{side}_gapcv']=cv
                a=asset[X.o]; mn=_query(_sparse(a,np.minimum),lo,hi,np.minimum,0); mx=_query(_sparse(a,np.maximum),lo,hi,np.maximum,0)
                out[f'burst_{side}_onecur']=(mn==mx).astype(float)*(cnt>=2)
                out[f'burst_{side}_newxcv']=(fs[hi]-fs[lo])/(1+np.maximum(cv,0))
    # slow chains (no value ratio), linking only through non-hub middle wallets
    for W in (1800,3600,7200,14400):
        okd=deg[dst]<=400; oks=deg[src]<=400
        lo,hi=S.rng(dst,t+1e-6,t+W); hi=np.where(okd,hi,lo)
        blo,bhi=R.rng(src,t-W,t-1e-6); bhi=np.where(oks,bhi,blo)
        fw=np.ones(n); bw=np.ones(n)
        for _ in range(8):
            fw=np.maximum(fw,1+np.where(hi>lo,_query(_sparse(fw[S.o],np.maximum),lo,hi,np.maximum,0),0))
            bw=np.maximum(bw,1+np.where(bhi>blo,_query(_sparse(bw[R.o],np.maximum),blo,bhi,np.maximum,0),0))
        out[f'slow_chain_{W}']=fw+bw-1
    # hub-pruned components
    for N in (200,400):
        for L in (3600,7200):
            rows=[];cols=[]
            for X,key in ((S,src),(R,dst)):
                ok=(X.k[1:]==X.k[:-1])&(np.diff(X.t)<=L)&(deg[X.k[1:]]<=N)
                rows.append(X.o[1:][ok]); cols.append(X.o[:-1][ok])
            r=np.concatenate(rows); c=np.concatenate(cols)
            _,lab=connected_components(coo_matrix((np.ones(len(r)),(r,c)),shape=(n,n)),directed=False)
            g=pd.DataFrame(dict(c=lab,s=src,d=dst,t=t,p=pair))
            gb=g.groupby('c'); size=gb.size(); ns=gb.s.nunique(); nr=gb.d.nunique(); ne=gb.p.nunique()
            w=pd.concat([g[['c','s']].rename(columns={'s':'w'}).assign(r=1),g[['c','d']].rename(columns={'d':'w'}).assign(r=2)])
            wr=w.groupby(['c','w']).r.agg(lambda x: x.nunique() if False else 0) if False else None
            sw=w.drop_duplicates(['c','w','r']).groupby(['c','w']).size()
            both=(sw==2).groupby(level=0).mean()
            g=g.sort_values(['c','t']); gap=g.t.diff().where(g.c.eq(g.c.shift()))
            gg=gap.groupby(g.c); med=gg.median(); cvv=gg.std()/(gg.mean()+1)
            tag=f'{N}_{L}'
            for nm,sr in (('size',size),('snd',ns),('rcv',nr),('dens',ne/(ns*nr)),('both',both),('medgap',med),('gapcv',cvv)):
                out[f'hub_{nm}_{tag}']=sr.reindex(lab).fillna(-1).to_numpy(float)
    base=pd.DataFrame(out)
    return base, src, dst, t, deg

def scenario_agg(base, cols, src, dst, t, deg, W=1800, N=400):
    n=len(base); res={}
    for X,key in ((Side(src,t),src),(Side(dst,t),dst)):
        lo,hi=X.rng(key,t-W,t+W); hi=np.where(deg[key]<=N,hi,lo+1)
        for c in cols:
            v=base[c].to_numpy(float)[X.o]; cs=np.concatenate([[0],np.cumsum(v)])
            mx=_query(_sparse(v,np.maximum),lo,hi,np.maximum,0); mn=(cs[hi]-cs[lo])/np.maximum(hi-lo,1)
            res[f'{c}_amax']=np.maximum(res.get(f'{c}_amax',-np.inf),mx); res[f'{c}_amean']=np.maximum(res.get(f'{c}_amean',-np.inf),mn)
    return pd.DataFrame(res)

AGG_COLS=['slow_chain_1800','burst_out_gapcv','slow_fout_cv_7200','slow_fin_cv_3600','burst_in_onecur','hub_both_400_7200']
def hub_feats(d):
    base,src,dst,t,deg=hub_features(d)
    if AGG_COLS: base=pd.concat([base,scenario_agg(base,AGG_COLS,src,dst,t,deg)],axis=1)
    return base
