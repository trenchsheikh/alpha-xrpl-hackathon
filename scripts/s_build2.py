import pandas as pd,numpy as np,hashlib,zipfile,glob
C='/workspace/crackit/';B=pd.read_csv('out/sub/q4_blend.csv')
base=B.sort_values('score',ascending=False).TRANSACTION_HASH.tolist();assert len(base)==40101==len(set(base))
def load(f,c,d=C):
  x=pd.read_csv(d+f).drop_duplicates('TRANSACTION_HASH').set_index('TRANSACTION_HASH'); s=x[c].astype(float).fillna(-1)
  if c!='score': s=s+1e-3*x['score'].fillna(0)
  return s.reindex(base).fillna(-1e18)
v4=load('hard_test_v4.csv','score');rl=load('rb2_test.csv','relaylike');ch=load('s_chain_test.csv','score','out/sub/')
top=lambda s,ex,K:[h for h in s.sort_values(ascending=False,kind='mergesort').index if h not in ex][:K]
V=[(9,'rb2_relaylike',[(rl,25)],150,'K25 0.08/K50 0.04'),(10,'rb2_relaylike',[(rl,25)],400,'K25 0.08/K50 0.04'),
(11,'hard_v4',[(v4,100)],150,'K100 0.06/K300 0.043'),(12,'hard_v4',[(v4,100)],400,'K100 0.06/K300 0.043'),
(13,'chain_c_gmax',[(ch,25)],300,'K25 0.00/K50 0.04/K100 0.02 (borderline)'),(14,'chain_c_gmax',[(ch,50)],300,'K25 0.00/K50 0.04/K100 0.02 (borderline)'),
(15,'relay25+hard_v4_75',[(rl,25),(v4,75)],300,'relay K25 0.08; v4 K100 0.06'),(16,'relay25+hard_v4_75',[(rl,25),(v4,75)],300,'')]
V=V[:7]
old={hashlib.md5(open(f,'rb').read()).hexdigest() for f in glob.glob('out/sub/*.csv')}
man=open('out/sub/s_manifest.txt','a')
for n,name,sigs,P,pr in V:
  ex=set(base[:P]);pick=[]
  for s,K in sigs: p=top(s,ex,K);pick+=p;ex|=set(p)
  ps=set(pick);order=base[:P]+pick+[h for h in base[P:] if h not in ps]
  assert len(order)==40101==len(set(order))
  f=f'out/sub/s{n}_hard.csv';pd.DataFrame({'TRANSACTION_HASH':order,'score':np.linspace(1,0,len(order))}).to_csv(f,index=False)
  m=hashlib.md5(open(f,'rb').read()).hexdigest();assert m not in old;old.add(m)
  with zipfile.ZipFile(f'out/sub/code_s{n}_hard.zip','w',zipfile.ZIP_DEFLATED) as z:
    for g in ('s_build2.py','s_chain.py'): z.write('out/sub/'+g,g)
  man.write(f'{f}\t{name} [base q4, depth {P}]\t{sum(k for _,k in sigs)}\t{pr}\t{m}\n')
