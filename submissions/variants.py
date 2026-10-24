import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
from grouped_cv import groups
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
T = 4
RELAY = ["relay_match6h", "relay_amount_matching"]
V = {
 "v01_motif": dict(desc="motif feats; motif features; CatBoost s42 best config (no relay, drop hard pos)", m=[("cat",42,{})]),
 "v02_cat3seeds": dict(desc="motif feats; rank-avg CatBoost seeds 42/7/99, best config (no relay, drop hard pos)", m=[("cat",42,{}),("cat",7,{}),("cat",99,{})]),
 "v03_cat1200": dict(desc="motif feats; CatBoost s42 1200 iters, best config", m=[("cat",42,{"iterations":1200})]),
 "v04_pseudo": dict(desc="motif feats; CatBoost s42 best config + pseudo-labels (test top 0.5% ->1, bottom 50% ->0) from full model", m=[("cat",42,{})], pseudo=True),
 "v05_cat75lgb25": dict(desc="motif feats; 0.75 CatBoost s42 / 0.25 LightGBM s42 rank blend, best config", m=[("cat",42,{}),("lgb",42,{})], w=[.75,.25]),
 "v06_relaykept": dict(desc="motif feats; CatBoost s42, relay features kept, hard pos dropped", m=[("cat",42,{})], relay=True),
 "v07_depth8": dict(desc="motif feats; CatBoost s42 depth 8, best config", m=[("cat",42,{"depth":8})]),
 "v08_classw": dict(desc="motif feats; CatBoost s42 scale_pos_weight=5, best config", m=[("cat",42,{"scale_pos_weight":5})]),
 "v09_cat5seeds": dict(desc="motif feats; rank-avg CatBoost seeds 42/7/99/1/2024, best config", m=[("cat",s,{}) for s in (42,7,99,1,2024)]),
}
def mk(k, s, p):
    if k == "cat":
        a = dict(iterations=600, depth=7, learning_rate=.04, l2_leaf_reg=6, random_seed=s, verbose=False, thread_count=T, allow_writing_files=False); a.update(p)
        return CatBoostClassifier(**a)
    return LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=31, min_child_samples=30, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=5, n_jobs=T, random_state=s, verbose=-1)
def fp(Xa, ya, Xb, v):
    ps = [mk(k,s,p).fit(Xa, ya).predict_proba(Xb)[:,1] for k,s,p in v["m"]]
    w = np.array(v.get("w") or [1]*len(ps), float)
    r = sum(wi*rankdata(p)/len(p) for wi,p in zip(w,ps))/w.sum() + 1e-3*np.mean(ps,0)
    return (r-r.min())/(r.max()-r.min()+1e-12)
name = sys.argv[1]; v = V[name]
lab = pd.read_csv('data/train/labels.csv')
tr = pd.read_parquet('cache/feats_motif_train.parquet').merge(lab, on="TRANSACTION_HASH", validate="one_to_one")
te = pd.read_parquet('cache/test_feats_motif.parquet')
drop = [] if v.get("relay") else RELAY
cols = [c for c in te.columns if c != "TRANSACTION_HASH" and c not in drop and c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])]
keep = ~((tr.difficulty == "hard") & (tr.is_suspicious == 1)).to_numpy()
y = tr.is_suspicious.astype(int).to_numpy(); X = tr[cols]; Xt = te[cols]
def pseudo():
    s = pd.read_csv('out/sub/v01_motif.csv').set_index("TRANSACTION_HASH").score.reindex(te.TRANSACTION_HASH).to_numpy()
    r = rankdata(s)/len(s); m = (r >= .995) | (r <= .5)
    return Xt[m], (r[m] >= .995).astype(int)
def fit_full(Xa, ya):
    if v.get("pseudo"):
        px, py = pseudo(); Xa = pd.concat([Xa, px]); ya = np.r_[ya, py]
    return Xa, ya
# full
Xa, ya = fit_full(X[keep], y[keep])
s = fp(Xa, ya, Xt, v)
sub = pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": np.clip(s,0,1)})
assert len(sub)==40101 and sub.TRANSACTION_HASH.is_unique and sub.score.between(0,1).all()
sub.to_csv(f'out/sub/{name}.csv', index=False); print("wrote", name, flush=True)
import csv
d=pd.read_csv(f'out/sub/{name}.csv');assert list(d.columns)==["TRANSACTION_HASH","score"] and len(d)==40101 and d.TRANSACTION_HASH.is_unique and d.score.between(0,1).all()
csv.writer(open('out/sub/manifest.csv','a',newline='')).writerow([f"{name}.csv",v["desc"],""]);print("ok",name,flush=True)
