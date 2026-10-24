import pandas as pd, numpy as np, zipfile, os
D='out/sub'
base=pd.read_csv(f'{D}/h_avg3prop.csv')[['TRANSACTION_HASH']]
def r(n):
    s=pd.read_csv(f'{D}/{n}.csv').set_index('TRANSACTION_HASH')['score'].reindex(base.TRANSACTION_HASH)
    assert s.notna().all(); return s.rank(pct=True).values
A,R,E,C=r('g_aggwide-prop'),r('g_f_rankavg-top-prop'),r('g_e_hub-classw-prop'),r('g_c2-prop')
B={'m1':.35*A+.35*R+.3*E,'m2':.3*A+.3*R+.2*E+.2*C,'m3':.4*A+.3*R+.3*E,'m4':.5*r('k2_blend')+.5*r('k1_blend')}
for k,v in B.items():
    o=base.copy(); o['score']=v; p=f'{D}/{k}_blend.csv'; o.to_csv(p,index=False)
    with zipfile.ZipFile(f'{D}/code_{k}_blend.zip','w') as z: z.write(__file__,'m_blends.py')
