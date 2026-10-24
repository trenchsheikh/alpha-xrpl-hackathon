import pandas as pd
a=pd.read_csv('g_scenlink2-prop.csv').set_index('TRANSACTION_HASH')['score'].rank(pct=True)
b=pd.read_csv('k2_blend.csv').set_index('TRANSACTION_HASH')['score'].rank(pct=True).reindex(a.index)
for n,w in [('n1',0.5),('n2',0.7)]:
    s=(w*a+(1-w)*b); assert s.notna().all() and len(s)==40101
    s.rename('score').reset_index().to_csv(f'{n}_blend.csv',index=False)
