import pandas as pd, zipfile
r=lambda f: pd.read_csv(f).set_index('TRANSACTION_HASH')['score'].rank(pct=True)
S=r('g_scenlink2-prop.csv'); idx=S.index
K2=r('k2_blend.csv').reindex(idx); H=r('h_avg3prop.csv').reindex(idx)
E=r('e_scenlink2-classw.csv').reindex(idx); HUB=r('g_e_hub-classw-prop.csv').reindex(idx)
A=r('g_aggwide-prop.csv').reindex(idx); F=r('g_f_rankavg-top-prop.csv').reindex(idx)
V={'q1':0.6*S+0.4*K2,'q2':0.4*S+0.3*K2+0.3*H,'q3':0.5*S+0.2*A+0.15*F+0.15*HUB,'q4':0.4*S+0.1*E+0.5*K2}
for n,s in V.items():
    assert s.notna().all() and len(s)==40101
    s.rename('score').reset_index().to_csv(f'{n}_blend.csv',index=False)
    z=zipfile.ZipFile(f'code_{n}_blend.zip','w'); z.write('q_blend.py'); z.close()
