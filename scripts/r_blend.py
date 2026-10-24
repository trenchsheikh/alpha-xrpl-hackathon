import pandas as pd, zipfile
r=lambda f: pd.read_csv(f).set_index('TRANSACTION_HASH')['score'].rank(pct=True)
S=r('g_scenlink2-prop.csv'); idx=S.index
K2=r('k2_blend.csv').reindex(idx); P1=r('r1_p1prop.csv').reindex(idx)
V={'r2':pd.concat([S,K2],axis=1).max(axis=1),'r3':pd.concat([S,K2,P1],axis=1).max(axis=1)}
for n,s in V.items():
    s=s+1e-9*(S+K2)  # tie-break
    assert s.notna().all() and len(s)==40101
    s.rename('score').reset_index().to_csv(f'{n}_blend.csv',index=False)
for n in ['r1','r2','r3']:
    z=zipfile.ZipFile(f'code_{n}_blend.zip','w'); z.write('r_blend.py'); z.write('/workspace/crackit/prop_apply.py','prop_apply.py'); z.close()
