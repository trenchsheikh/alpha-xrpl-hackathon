import pandas as pd,numpy as np,collections,bisect
from sklearn.metrics import average_precision_score as AP
def detect(m,W,lo=0.4,hi=1.05):
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
            if lo<=r<=hi and d[j]!=s[i] or (lo<=r<=hi and d[j]==s[i]): nxt[i].append(j)
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
            if k<5:
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

def motif_features(d):
    """Label-free exact-motif detectors on (TRANSACTION_HASH,TIMESTAMP,FROM,TO,VALUE=USD)."""
    m=pd.DataFrame(dict(TRANSACTION_HASH=d.TRANSACTION_HASH.values,ts=d.TIMESTAMP.values.astype(float),
        src=d.FROM_ADDRESS.astype(str).values,dst=d.TO_ADDRESS.astype(str).values,u=d.VALUE.values.astype(float),sym=0))
    out=m[['TRANSACTION_HASH']]
    for W,cols in ((300,['chain','fin','fout','foi']),(900,['foi','chain']),(2700,['chain','cyc','foi'])):
        r=detect(m,W,0.75,1.2).set_index('TRANSACTION_HASH')
        for c in cols: out=out.assign(**{f'motif_{c}_{W}':m.TRANSACTION_HASH.map(r[c]).values})
    return out
