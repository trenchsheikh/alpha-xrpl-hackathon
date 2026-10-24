import pandas as pd,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score as ap
tr=pd.read_pickle('tr.pkl');te=pd.read_pickle('te.pkl')
F=['cx','lu','round','s_out','s_in','r_out','r_in','s_age','r_age','s_new','s_cv','r_cv','r_nsend','s_ndst','cyc','s_ncur']
S='dev-tester/alpha-xrpl-hackathon/out/sub/'
base=pd.read_csv(S+'n1_blend.csv').sort_values('score',ascending=False).reset_index(drop=True)
m=H(max_iter=200,learning_rate=.05,random_state=0)
tr['oofh']=cross_val_predict(m,tr[F],tr.hard,cv=5,method='predict_proba')[:,1]
m.fit(tr[F],tr.hard);te['mh']=m.predict_proba(te[F])[:,1]
te[['TRANSACTION_HASH','mh']].rename(columns={'mh':'score'}).to_csv(S+'hardmodel_test_scores.csv',index=False)
hard=te.set_index('TRANSACTION_HASH').mh.reindex(base.TRANSACTION_HASH).fillna(0).values
hs=pd.Series(hard,index=base.TRANSACTION_HASH).sort_values(ascending=False)
def chk(o,n):
    assert len(o)==40101 and o.TRANSACTION_HASH.is_unique and not o.isna().any().any();o.to_csv(S+n+'.csv',index=False)
notes=[]
for n,keep,k in [('h11_hardblend',150,60),('h12_hardblend',250,100),('h13_hardblend',350,150)]:
    excl=set(base.TRANSACTION_HASH[:400]) if n=='h11_hardblend' else set(base.TRANSACTION_HASH[:keep])
    c=[h for h in hs.index if h not in excl][:k];cs=set(c)
    rest=[h for h in base.TRANSACTION_HASH if h not in cs]
    order=rest[:keep]+c+rest[keep:]
    chk(pd.DataFrame({'TRANSACTION_HASH':order,'score':np.linspace(1,0,len(order))}),n)
    notes.append(f'{n}.csv: keep base top {keep}, insert {len(c)} hard at {keep+1}-{keep+len(c)}')
rb=pd.Series(base.score.values).rank(pct=True).values;rh=pd.Series(hard).rank(pct=True).values
chk(pd.DataFrame({'TRANSACTION_HASH':base.TRANSACTION_HASH,'score':np.maximum(rb,.8*rh)}),'h14_hardblend')
chk(pd.DataFrame({'TRANSACTION_HASH':base.TRANSACTION_HASH,'score':.85*rb+.15*rh}),'h15_hardblend')
o=pd.read_csv('dev-tester/alpha-xrpl-hackathon/out/cv/repo-agg-wide_oof.csv').merge(tr[['TRANSACTION_HASH','is_suspicious','hard','oofh']],on='TRANSACTION_HASH')
a=o.score.rank(pct=True);b=o.oofh.rank(pct=True);y=o.is_suspicious
r=f'OOF (repo-agg-wide, n={len(o)}): base AP={ap(y,o.score):.4f}; h14 max(rb,.8rh) AP={ap(y,np.maximum(a,.8*b)):.4f}; h15 .85/.15 AP={ap(y,.85*a+.15*b):.4f}; hard-only AP(susp)={ap(y,o.oofh):.4f}; hard-only AP(hard)={ap(o.hard,o.oofh):.4f}; hard prec@top200={o.sort_values("oofh").hard.iloc[-200:].mean():.3f}'
print(r);print('\n'.join(notes))
open('dev-tester/probe_notes.md','a').write('\n\n## h11-h15 hard-model blends (2026-10-24)\nBase: n1_blend.csv (cheese toastie, 66.0). Hard GBM retrained on all train hard labels (probe.py feats), test scores in hardmodel_test_scores.csv.\n'+'\n'.join('- '+x for x in notes)+'\n- h14_hardblend.csv: max(rank_base,0.8*rank_hard)\n- h15_hardblend.csv: 0.85 rank_base+0.15 rank_hard\n- '+r+'\nNot submitted.\n')
