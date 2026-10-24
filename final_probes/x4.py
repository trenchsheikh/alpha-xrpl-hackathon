exec(open('x1.py').read().split('for k in')[0])
m=H(max_iter=200,learning_rate=.05,random_state=0).fit(tr[F],T['med'].astype(int))
pm=pd.Series(m.predict_proba(te[F])[:,1],index=te.TRANSACTION_HASH).reindex(base.TRANSACTION_HASH).fillna(0).rank(pct=True).values
chk(pd.DataFrame({'TRANSACTION_HASH':base.TRANSACTION_HASH,'score':0.999*pm+0.001*rb}),'z2_medtop')
chk(pd.DataFrame({'TRANSACTION_HASH':base.TRANSACTION_HASH,'score':np.maximum(rb,.8*pm)}),'x4_medonly')
