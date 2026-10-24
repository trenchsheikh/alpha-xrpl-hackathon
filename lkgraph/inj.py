import pandas as pd,numpy as np,glob
from sklearn.metrics import average_precision_score as ap
def inject(b,m,K,pos):
  order=list(np.argsort(-b));top=set(order[:401])
  pick=[i for i in np.argsort(-m) if i not in top][:K];ps=set(pick)
  rest=[i for i in order if i not in ps];new=rest[:pos]+pick+rest[pos:]
  s=np.empty(len(b));s[new]=1-np.arange(len(b))/len(b);return s
if __name__=='__main__':
  R='/workspace/dev-tester/alpha-xrpl-hackathon/'
  l=pd.read_csv(R+'data/train/labels.csv');y=l.is_suspicious.values
  base=sum(pd.read_csv(f).score.rank(pct=True).values for f in sorted(glob.glob(R+'out/cv/repo-*_oof.csv')))
  o=pd.read_parquet('oof.parquet')['base+g'].values
  def rec(s):
    top=np.argsort(-s)[:401];return [round(((l.difficulty==d)&(y==1)).values[top].sum()/((l.difficulty==d)&(y==1)).sum(),3) for d in['easy','medium','hard']]
  print('base',round(ap(y,base),4),rec(base))
  for K in [10,20,40,80]:
    for pos in [200,380]:
      s=inject(base,o,K,pos);print(K,pos,round(ap(y,s),4),rec(s))
