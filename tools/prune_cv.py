"""In-fold importance pruning (no leakage): per fold fit CatBoost, keep top-K by importance, refit, predict."""
import sys; sys.path[:0] = ["tools", "."]
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from grouped_cv import groups, metrics, META
import ensemble_boost as eb
f = pd.read_parquet(sys.argv[1]); d = f.merge(pd.read_csv("data/train/labels.csv"), on="TRANSACTION_HASH")
X = d.drop(columns=["TRANSACTION_HASH"] + META).select_dtypes("number"); y = d.is_suspicious.to_numpy()
Ks = [int(k) for k in sys.argv[2:]]; cfg = dict(eb.DEFAULT); oof = {k: np.zeros(len(y)) for k in Ks}; fid = np.zeros(len(y), int)
for i, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups(d))):
    fid[va] = i; m = eb.make("cat", 42, cfg); m.fit(X.iloc[tr], y[tr]); imp = pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False)
    for k in Ks:
        cols = imp.index[:k]; m2 = eb.make("cat", 42, cfg); m2.fit(X.iloc[tr][cols], y[tr]); oof[k][va] = m2.predict_proba(X.iloc[va][cols])[:, 1]
    print("fold", i, "done", flush=True)
for k in Ks:
    r = metrics(d, oof[k], fid); np.save(f"out/ens/cat42_top{k}.npy", oof[k])
    print(f"top{k} AP {r['AP_x100']} folds {r['fold_AP_x100']} med R@1% {r['by_kind']['medium']['recall@1%']} medAP {r['by_kind']['medium']['AP_vs_neg_x100']}", flush=True)
