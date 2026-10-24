import pandas as pd,numpy as np,json
from sklearn.ensemble import HistGradientBoostingClassifier as H
R={'XRP':0.648,'USD':1,'GBP':1.30,'EUR':1.10}
def feats(d):
    t=pd.read_parquet(f'xrpl-data/{d}/transactions.parquet')
    t=t[['TRANSACTION_HASH','ACCOUNT','TIMESTAMP','EXTRA_TRANSACTION_FIELDS']].copy()
    t['ts']=t.TIMESTAMP.astype(float)
    def p(s):
        try:j=json.loads(s)
        except:return (None,'XRP',0.0,'XRP')
        a=j.get('Amount');sm=j.get('SendMax')
        if isinstance(a,dict):c,v=a.get('currency'),float(a.get('value',0))
        elif a is not None:c,v='XRP',float(a)/1e6
        else:c,v='XRP',0.0
        sc=sm.get('currency') if isinstance(sm,dict) else ('XRP' if sm is not None else c)
        return (j.get('Destination'),c,v,sc)
    P=t.EXTRA_TRANSACTION_FIELDS.map(p)
    t['dst']=[x[0] for x in P];t['cur']=[x[1] for x in P];t['scur']=[x[3] for x in P]
    t['usd']=[x[2]*R.get(x[1],1) for x in P]
    t['cx']=(t.cur!=t.scur).astype(int)
    t['lu']=np.log1p(t.usd)
    t['round']=((t.usd%10<0.5)|(t.usd%10>9.5)).astype(int)&(t.usd>=50)
    t['round']=((np.abs(t.usd-np.round(t.usd,-1))<0.05*1)&(t.usd>=50)).astype(int)
    od=t.groupby('ACCOUNT').size();idg=t.groupby('dst').size()
    t['s_out']=t.ACCOUNT.map(od).fillna(0);t['s_in']=t.ACCOUNT.map(idg).fillna(0)
    t['r_out']=t.dst.map(od).fillna(0);t['r_in']=t.dst.map(idg).fillna(0)
    first=pd.concat([t[['ACCOUNT','ts']].rename(columns={'ACCOUNT':'a'}),t[['dst','ts']].rename(columns={'dst':'a'})]).groupby('a').ts.min()
    t['s_age']=t.ts-t.ACCOUNT.map(first);t['r_age']=t.ts-t.dst.map(first)
    t['s_new']=(t.s_age<86400).astype(int)
    # split equality: cv of amounts sent by sender
    g=t.groupby('ACCOUNT').usd;t['s_cv']=t.ACCOUNT.map(g.std()/g.mean()).fillna(9)
    g2=t.groupby('dst').usd;t['r_cv']=t.dst.map(g2.std()/g2.mean()).fillna(9)
    t['r_nsend']=t.dst.map(t.groupby('dst').ACCOUNT.nunique()).fillna(0)
    t['s_ndst']=t.ACCOUNT.map(t.groupby('ACCOUNT').dst.nunique()).fillna(0)
    # cycles: 2-3 hop return
    E=set(zip(t.ACCOUNT,t.dst));nb=t.groupby('ACCOUNT').dst.apply(set).to_dict()
    def cyc(a,b):
        if (b,a) in E:return 1
        return int(any((c,a) in E for c in nb.get(b,())))
    t['cyc']=[cyc(a,b) for a,b in zip(t.ACCOUNT,t.dst)]
    t['s_ncur']=t.ACCOUNT.map(t.groupby('ACCOUNT').cur.nunique())
    return t
F=['cx','lu','round','s_out','s_in','r_out','r_in','s_age','r_age','s_new','s_cv','r_cv','r_nsend','s_ndst','cyc','s_ncur']
tr=feats('train').merge(pd.read_csv('xrpl-data/train/labels.csv'),on='TRANSACTION_HASH')
te=feats('test')
tr['hard']=((tr.is_suspicious==1)&(tr.difficulty=='hard')).astype(int)
tr.to_pickle('tr.pkl');te.to_pickle('te.pkl');print(tr.hard.sum())
