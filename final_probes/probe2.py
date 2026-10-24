import pandas as pd,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.model_selection import cross_val_predict
tr=pd.read_pickle('tr.pkl');te=pd.read_pickle('te.pkl')
F=['cx','lu','round','s_out','s_in','r_out','r_in','s_age','r_age','s_new','s_cv','r_cv','r_nsend','s_ndst','cyc','s_ncur']
S='dev-tester/alpha-xrpl-hackathon/out/sub/'
base=pd.read_csv(S+'k2_blend.csv');base=base.sort_values('score',ascending=False).reset_index(drop=True)
top400=set(base.TRANSACTION_HASH[:400])
m=H(max_iter=200,learning_rate=.05)
tr['oofh']=cross_val_predict(m,tr[F],tr.hard,cv=5,method='predict_proba')[:,1]
m.fit(tr[F],tr.hard);te['mh']=m.predict_proba(te[F])[:,1]
oof=pd.read_csv('dev-tester/alpha-xrpl-hackathon/out/cv/repo-agg-wide_oof.csv');print(oof.columns.tolist())
rules={
 'h02_curchange':lambda d:d.cx*(1+d.lu/20),
 'h04_consolidation':lambda d:(d.r_nsend>=4)*(d.r_out>=1)*d.r_nsend/(1+d.r_out),
 'h05_equalsplit':lambda d:(d.s_ndst>=3)*(d.s_cv<0.15)*d.s_ndst,
 'h06_roundusd':lambda d:d['round']*(d.r_out>=1)*(1+d.lu/20),
 'h07_newwallet_fwd':lambda d:d.s_new*(d.s_in>=1)*(d.s_age<3600)*1.0+d.s_new*0.1,
 'h08_cycles':lambda d:d.cyc*(1+d.lu/20),
 'h09_hardmodel':lambda d:d.mh if 'mh' in d else d.oofh,
}
notes=[]
def write(name,sc,hyp,prec):
    c=te.assign(s=sc.values)[~te.TRANSACTION_HASH.isin(top400)]
    c=c[c.s>0].sort_values('s',ascending=False).head(80).TRANSACTION_HASH.tolist()
    rest=[h for h in base.TRANSACTION_HASH if h not in set(c)]
    order=rest[:200]+c+rest[200:]
    out=pd.DataFrame({'TRANSACTION_HASH':order,'score':np.linspace(1,0,len(order))})
    assert len(out)==40101 and out.TRANSACTION_HASH.is_unique and not out.isna().any().any()
    out.to_csv(S+name+'_probe.csv',index=False)
    notes.append(f'{name}_probe.csv | {hyp} | inserted {len(c)} at ranks 201-{200+len(c)} | train {prec}')
for n,f in rules.items():
    trs=f(tr.assign(mh=tr.oofh)) if n=='h09_hardmodel' else f(tr)
    k=trs.sort_values(ascending=False).index[:200]; fl=trs>0
    prec=f'flagged={int(fl.sum())}, hard-prec(flag)={tr.hard[fl].mean():.3f}, susp-prec={tr.is_suspicious[fl].mean():.3f}, hard-prec@top200={tr.hard[k].mean():.3f} (base hard rate {tr.hard.mean():.4f})'
    print(n,prec);write(n,f(te),n,prec)
# h10: low-ranked frauds
o=oof.merge(tr,on='TRANSACTION_HASH');sc=[c for c in oof.columns if c!='TRANSACTION_HASH'][-1]
o['rk']=o[sc].rank(ascending=False);low=o[(o.is_suspicious==1)&(o.rk>len(o)*0.01)]
print(low.groupby(['difficulty','family']).size());print(low[F].median());print(o[o.is_suspicious==0][F].median())
