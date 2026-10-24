"""SENTINEL V1: full-window transaction and wallet behavioural CatBoost ranking."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from payments import load_payments
SEED=42

def build_features(folder):
 d=pd.DataFrame(load_payments(folder))
 if not len(d): raise RuntimeError("Empty payments")
 d=d.sort_values(["TIMESTAMP","TRANSACTION_HASH"]).reset_index(drop=True)
 d["VALUE"]=pd.to_numeric(d.VALUE,errors="coerce").fillna(0).clip(lower=0)
 d["TIMESTAMP"]=pd.to_numeric(d.TIMESTAMP)
 o=pd.DataFrame({"TRANSACTION_HASH":d.TRANSACTION_HASH})
 o["log_usd"]=np.log1p(d.VALUE)
 o["hour"]=(d.TIMESTAMP//3600)%24
 o["time_fraction"]=(d.TIMESTAMP-d.TIMESTAMP.min())/max(1,d.TIMESTAMP.max()-d.TIMESTAMP.min())
 o["self_payment"]=(d.FROM_ADDRESS==d.TO_ADDRESS).astype(int)
 send=d.groupby("FROM_ADDRESS").agg(send_count=("VALUE","size"),sent_usd=("VALUE","sum"),send_mean=("VALUE","mean"),send_std=("VALUE","std"),recipients=("TO_ADDRESS","nunique"))
 recv=d.groupby("TO_ADDRESS").agg(recv_count=("VALUE","size"),received_usd=("VALUE","sum"),recv_mean=("VALUE","mean"),recv_std=("VALUE","std"),senders=("FROM_ADDRESS","nunique"))
 for side,account in [("s",d.FROM_ADDRESS),("r",d.TO_ADDRESS)]:
  for name,col in send.items(): o[side+"_"+name]=account.map(col).fillna(0).to_numpy()
  for name,col in recv.items(): o[side+"_"+name]=account.map(col).fillna(0).to_numpy()
 o["s_balance_logratio"]=np.log1p(o.s_sent_usd)-np.log1p(o.s_received_usd)
 o["r_balance_logratio"]=np.log1p(o.r_sent_usd)-np.log1p(o.r_received_usd)
 o["relative_sender_usd"]=d.VALUE/(o.s_send_mean+1)
 o["relative_receiver_usd"]=d.VALUE/(o.r_recv_mean+1)
 pair=d.groupby(["FROM_ADDRESS","TO_ADDRESS"]).size()
 o["pair_count"]=[pair.get((a,b),0) for a,b in zip(d.FROM_ADDRESS,d.TO_ADDRESS)]
 o["reverse_pair_count"]=[pair.get((b,a),0) for a,b in zip(d.FROM_ADDRESS,d.TO_ADDRESS)]
 sent={a:(g.TIMESTAMP.to_numpy(),g.VALUE.to_numpy()) for a,g in d.groupby("FROM_ADDRESS")}
 received={a:(g.TIMESTAMP.to_numpy(),g.VALUE.to_numpy()) for a,g in d.groupby("TO_ADDRESS")}
 times=d.TIMESTAMP.to_numpy(); amounts=d.VALUE.to_numpy()
 for label,addresses,buckets,look_forward in [
  ("sender_prior_in",d.FROM_ADDRESS,received,False),
  ("receiver_next_out",d.TO_ADDRESS,sent,True),
  ("sender_next_out",d.FROM_ADDRESS,sent,True),
  ("receiver_prior_in",d.TO_ADDRESS,received,False)]:
  count=np.zeros(len(d)); similar=np.zeros(len(d)); delay=np.full(len(d),3601.)
  for i,(addr,t,value) in enumerate(zip(addresses,times,amounts)):
   if addr not in buckets:continue
   ts,vals=buckets[addr]
   if look_forward:
    start=np.searchsorted(ts,t,side="right");end=np.searchsorted(ts,t+3600,side="right")
    if start<len(ts):delay[i]=min(3601.,ts[start]-t)
   else:
    start=np.searchsorted(ts,t-3600,side="left");end=np.searchsorted(ts,t,side="left")
    if end>0:delay[i]=min(3601.,t-ts[end-1])
   count[i]=end-start
   if end>start and value>0:
    vv=vals[start:end]
    similar[i]=float(np.max(np.minimum(vv,value)/(np.maximum(vv,value)+1e-9)))
  o[label+"_count1h"]=count
  o[label+"_amount_match"]=similar
  o[label+"_delay_sec"]=delay
 o["relay_pair_match"]=o.sender_prior_in_amount_match*o.receiver_next_out_amount_match
 return o.replace([np.inf,-np.inf],np.nan).fillna(0)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--train",default="data/train")
 p.add_argument("--test",default="data/test")
 p.add_argument("--output",default="predictions.csv")
 a=p.parse_args()
 train=build_features(a.train)
 labels=pd.read_csv(Path(a.train)/"labels.csv")
 assert labels.TRANSACTION_HASH.is_unique
 assert train.TRANSACTION_HASH.is_unique
 train=train.merge(labels[["TRANSACTION_HASH","is_suspicious"]],on="TRANSACTION_HASH",validate="one_to_one")
 X=train.drop(columns=["TRANSACTION_HASH","is_suspicious"])
 y=train.is_suspicious.astype(int)
 ids=np.arange(len(y))
 itr,iv=train_test_split(ids,test_size=.2,stratify=y,random_state=SEED)
 params=dict(iterations=1100,depth=7,learning_rate=.045,l2_leaf_reg=5,loss_function="Logloss",eval_metric="PRAUC",random_seed=SEED,verbose=False,thread_count=-1)
 model=CatBoostClassifier(**params)
 model.fit(X.iloc[itr],y.iloc[itr],eval_set=(X.iloc[iv],y.iloc[iv]),early_stopping_rounds=100)
 score=average_precision_score(y.iloc[iv],model.predict_proba(X.iloc[iv])[:,1])
 print("Train rows",len(train),"positives",int(y.sum()),"features",len(X.columns))
 print("Held-out AP:",score,"competition-scale:",100*score)
 print("Top features", sorted(zip(X.columns,model.feature_importances_),key=lambda pair:-pair[1])[:15])
 params["iterations"]=max(100,model.best_iteration_+1)
 final=CatBoostClassifier(**params)
 final.fit(X,y)
 test=build_features(a.test)
 T=test.reindex(columns=["TRANSACTION_HASH",*X.columns],fill_value=0)
 scores=final.predict_proba(T[X.columns])[:,1]
 submission=pd.DataFrame({"TRANSACTION_HASH":T.TRANSACTION_HASH,"score":np.clip(scores,0,1)})
 assert submission.TRANSACTION_HASH.is_unique and submission.score.between(0,1).all()
 if Path(a.test).name=="test": assert len(submission)==40101, "Expected official 40,101 test transactions"
 submission.to_csv(a.output,index=False)
 final.save_model(str(Path(a.output).with_suffix(".cbm")))
 print("Wrote",a.output,len(submission),"rows")
if __name__=="__main__":main()
