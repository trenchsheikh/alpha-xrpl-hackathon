import pandas as pd,numpy as np,hashlib,zipfile,glob,os
C='/workspace/crackit/';B=pd.read_csv('out/sub/n1_blend.csv')
base=B.sort_values('score',ascending=False).TRANSACTION_HASH.tolist()
def load(f,c):
  d=pd.read_csv(C+f).set_index('TRANSACTION_HASH'); s=d[c].astype(float).fillna(-1)
  if c!='score': s=s+1e-3*d['score'].fillna(0)
  return s.reindex(base).fillna(-1e9)
v4=load('hard_test_v4.csv','score');rl=load('rb2_test.csv','relaylike');v1=load('hard_test.csv','score')
mix=(v4.rank(pct=True)+rl.rank(pct=True))/2
prec={'hard_v4':'K25 0.00/K50 0.00/K100 0.06/K300 0.043','rb2_relaylike':'K25 0.08/K50 0.04/K100 0.02','hard_v1':'K25 0.04/K50 0.02/K100 0.02','mix_v4_relay':'n/a (rank avg)'}
V=[(1,'hard_v4',v4,25),(2,'hard_v4',v4,50),(3,'hard_v4',v4,100),(4,'rb2_relaylike',rl,25),(5,'rb2_relaylike',rl,50),(6,'hard_v1',v1,25),(7,'hard_v1',v1,50),(8,'mix_v4_relay',mix,50)]
old={hashlib.md5(open(f,'rb').read()).hexdigest() for f in glob.glob('out/sub/*.csv')}
man=open('out/sub/s_manifest.txt','w');man.write('file\tsignal\tK\ttrain_prec_hard+ambig(top-K on train; base rate 0.0056; train has 0 ambiguous)\tmd5\n')
P=300
for n,name,s,K in V:
  top=set(base[:P]);pick=[h for h in s.sort_values(ascending=False).index if h not in top][:K]
  ps=set(pick);rest=[h for h in base[P:] if h not in ps];order=base[:P]+pick+rest
  assert len(order)==40101==len(set(order))
  out=pd.DataFrame({'TRANSACTION_HASH':order,'score':np.linspace(1,0,len(order))})
  f=f'out/sub/s{n}_hard.csv';out.to_csv(f,index=False)
  m=hashlib.md5(open(f,'rb').read()).hexdigest();assert m not in old;old.add(m)
  with zipfile.ZipFile(f'out/sub/code_s{n}_hard.zip','w',zipfile.ZIP_DEFLATED) as z: z.write('out/sub/s_build.py','s_build.py')
  man.write(f'{f}\t{name}\t{K}\t{prec[name]}\t{m}\n')
man.close()
