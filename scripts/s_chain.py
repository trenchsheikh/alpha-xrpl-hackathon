import pandas as pd,numpy as np
src=open('/workspace/jcg/chain.py').read().split("FE=[")[0]; exec(src)
L=pd.read_csv('/workspace/xrpl-data/train/labels.csv').set_index('TRANSACTION_HASH');ha=L.difficulty.isin(['hard','ambiguous'])
tr=feats('train').drop_duplicates('TRANSACTION_HASH').set_index('TRANSACTION_HASH').reindex(L.index)
te=feats('test').drop_duplicates('TRANSACTION_HASH').set_index('TRANSACTION_HASH')
best=[]
for c in tr.columns:
  for sg in (1,-1):
    s=(sg*tr[c]).fillna(-1e18); o=s.sort_values(ascending=False,kind='mergesort').index
    p=[ha[o[:k]].mean() for k in (25,50,100)]; best.append((p[1],c,sg,p))
best.sort(reverse=True)
for b in best[:6]: print(b)
p,c,sg,_=best[0]; (sg*te[c]).rename('score').to_csv('/tmp/chain_test.csv'); print('chosen',c,sg)
