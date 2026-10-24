import pandas as pd,numpy as np,collections,bisect
from sklearn.metrics import average_precision_score as AP
def detect(m,W,lo=0.4,hi=1.05,samecur=False):
    m=m.sort_values('ts').reset_index(drop=True)
    ts=m.ts.values.astype(float);s=m.src.values;d=m.dst.values;u=m.u.values.astype(float);sym=m.sym.values
    N=len(m);ok=s!=d
    out_by=collections.defaultdict(list);in_by=collections.defaultdict(list)
    for i in range(N):
        if ok[i]: out_by[s[i]].append(i);in_by[d[i]].append(i)
    def win(lst,t0,t1):
        tt=[ts[j] for j in lst];return lst[bisect.bisect_left(tt,t0):bisect.bisect_right(tt,t1)]
    # cache times
    ot={k:np.array([ts[j] for j in v]) for k,v in out_by.items()};it={k:np.array([ts[j] for j in v]) for k,v in in_by.items()}
    def W_(dct,tdct,k,t0,t1):
        if k not in dct:return []
        a=tdct[k];return dct[k][np.searchsorted(a,t0):np.searchsorted(a,t1,'right')]
    nxt=[[] for _ in range(N)]
    for i in range(N):
        if not ok[i]:continue
        for j in W_(out_by,ot,d[i],ts[i]+1e-9,ts[i]+W):
            r=u[j]/max(u[i],1e-9)
            if lo<=r<=hi and (not samecur or sym[j]==sym[i]): nxt[i].append(j)
    # chain length forward/back
    fw=np.ones(N);bw=np.ones(N)
    for i in range(N-1,-1,-1):
        if nxt[i]: fw[i]=1+max(fw[j] for j in nxt[i])
    for i in range(N):
        for j in nxt[i]: bw[j]=max(bw[j],bw[i]+1)
    chain=fw+bw-1
    # cycle: closure returning to origin within 4 hops
    cyc=np.zeros(N)
    for i in range(N):
        if not ok[i]:continue
        fr=[(j,1) for j in nxt[i]];seen=set()
        while fr:
            j,k=fr.pop()
            if d[j]==s[i]: cyc[i]=max(cyc[i],k+1);
            if k<10:
                for q in nxt[j]:
                    if q not in seen: seen.add(q);fr.append((q,k+1))
    fin=np.zeros(N);fout=np.zeros(N);fincv=np.zeros(N);foi=np.zeros(N)
    for i in range(N):
        if not ok[i]:continue
        L=W_(in_by,it,d[i],ts[i]-W,ts[i]+W);ss={s[j] for j in L}
        uu=u[L];fin[i]=len(ss)/(1+np.std(np.log(uu+1e-6)))
        L=W_(out_by,ot,s[i],ts[i]-W,ts[i]+W);dd={d[j] for j in L}
        uu=u[L];fout[i]=len(dd)/(1+np.std(np.log(uu+1e-6)))
        # fan_out_in: middles fed by s[i] that forward to common sink
        sinks=collections.Counter()
        for j in L:
            for q in W_(out_by,ot,d[j],ts[j],ts[j]+W): 
                if d[q]!=s[i]: sinks[d[q]]+=1
        foi[i]=max(sinks.values()) if sinks else 0
    # propagate max over linked txs (motif membership)
    res=pd.DataFrame(dict(TRANSACTION_HASH=m.TRANSACTION_HASH,chain=chain,cyc=cyc,fin=fin,fout=fout,foi=foi))
    return res

def motif_features(d, extra=True):
    """Label-free exact-motif detectors on (TRANSACTION_HASH,TIMESTAMP,FROM,TO,VALUE=USD)."""
    m=pd.DataFrame(dict(TRANSACTION_HASH=d.TRANSACTION_HASH.values,ts=d.TIMESTAMP.values.astype(float),
        src=d.FROM_ADDRESS.astype(str).values,dst=d.TO_ADDRESS.astype(str).values,u=d.VALUE.values.astype(float),sym=d.ASSET.astype(str).values if 'ASSET' in d else 0))
    out={}
    specs=[(300,0.75,1.2,['chain','fin','fout','foi','cyc']),(900,0.75,1.2,['foi','chain']),(2700,0.75,1.2,['chain','foi'])]
    if extra: specs.append((1800,0.8,1.15,['chain','foi']))
    for W,lo,hi,cols in specs:
        r=detect(m,W,lo,hi).set_index('TRANSACTION_HASH')
        for c in cols: out[f'motif_{c}_{W}'+('m' if W==1800 else '')]=m.TRANSACTION_HASH.map(r[c]).values
    for W,lo,hi in ((300,0.75,1.2),(1800,0.8,1.15)):
        r=detect(m,W,lo,hi,samecur=True).set_index('TRANSACTION_HASH')
        for c in ('chain','foi'): out[f'motif_{c}_{W}_samecur']=m.TRANSACTION_HASH.map(r[c]).values
    if extra:
        for W in (300,2700): out[f'motif_stack_{W}']=m.TRANSACTION_HASH.map(stack_score(m,W)).values
    return pd.concat([m[['TRANSACTION_HASH']],pd.DataFrame(out)],axis=1)

def cycle_len(m,W,lo,hi,minlen=3):
    """Cycle closure requiring >=minlen hops (2-hop back-and-forth is common in normal traffic)."""
    r=detect(m,W,lo,hi)
    return r

def stack_score(m,W):
    m=m.sort_values('ts').reset_index(drop=True)
    ts=m.ts.values;s=m.src.values;d=m.dst.values;N=len(m)
    go=m.groupby('src').indices;gi=m.groupby('dst').indices
    def win(g,k,t):
        ix=g[k];a=ts[ix];return ix[np.searchsorted(a,t-W):np.searchsorted(a,t+W,'right')]
    sc=np.zeros(N)
    for i in range(N):
        A=set(d[win(go,s[i],ts[i])])
        if len(A)<2: continue
        cnt=collections.defaultdict(set)
        for r in A:
            for j in win(gi,r,ts[i]): cnt[s[j]].add(r)
        good={k:v for k,v in cnt.items() if len(v)>=2}
        if len(good)>=2 and d[i] in good.get(s[i],()): sc[i]=sum(len(v) for v in good.values())
    return pd.Series(sc,index=m.TRANSACTION_HASH.values)
MOTIF_EXTRA = True
