"""Temporal money-flow and bubble-map algorithms on synthetic toy chains."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from features_v3 import features_from_edges_v3
from flow_v31 import add_flow_features, risk_diffusion
from train_v31 import possible_scores, difficulty_metrics

transactions=[
 ("a",100,"A","B",60),
 ("b",110,"K","B",39),
 ("c",150,"B","C",99),
 ("d",180,"C","D",98),
 ("e",210,"D","E",97),
 ("f",250,"E","F",96),
 ("g",300,"X","Y",5),
 ("h",400,"P","P",3),
 ("i",600,"Y","Z",4),
 ("j",620,"Q","R",1),
 ("k",650,"Q","T",2),
]
data=pd.DataFrame(transactions,columns=[
 "TRANSACTION_HASH","TIMESTAMP","FROM_ADDRESS","TO_ADDRESS","VALUE"])
base=features_from_edges_v3(data)
feat,nxt,prv=add_flow_features(data,base)
assert len(feat)==len(data)
assert len(feat.columns)>len(base.columns)+20
assert np.isfinite(feat.drop(columns="TRANSACTION_HASH").to_numpy()).all()
index=feat.set_index("TRANSACTION_HASH")
assert index.loc["c","fundflow_forward_3"]>0
assert index.loc["c","prior_input_pair_3600"]>.95
assert index.loc["c","fundflow_matches_previous"]>=2
assert index.loc["d","fundflow_continuity_2way"]>0
seeds=np.array([.1,.2,.99,.98,.9,.85,.02,.01,.03,.1,.01])
support=risk_diffusion(seeds,nxt,prv)
assert np.isfinite(support).all() and support.min()>=0 and support.max()<=1
def pred(x): return {k:x.copy() for k in ("v3_cat","v3_lgb","flow_cat","flow_lgb")}
options=possible_scores(pred(seeds),nxt,prv,total_n=len(data))
assert "mix_75_trace_0.12" in options
assert all(len(v)==len(data) and np.isfinite(v).all() for v in options.values())
sliced=possible_scores(pred(seeds[2:6]),nxt,prv,
                       valid_indices=np.arange(2,6),total_n=len(data))
assert len(sliced["mix_75_trace_0.12"])==4
metrics,hard_ap=difficulty_metrics(
 np.array(["easy","hard","medium","hard","easy","hard","medium","easy","medium","hard","hard"]),
 (seeds>.95).astype(int),seeds)
assert "hard" in metrics and hard_ap >=0
print("SENTINEL V3.1 chain, subset, graph propagation and validation test PASSED")
