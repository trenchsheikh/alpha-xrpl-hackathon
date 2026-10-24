"""c-batch: features_v3.build_features_v3 -> 0.75 CatBoost / 0.25 LightGBM percentile-rank blend.
CatBoost: seeds 42/7/99 rank-averaged, scale_pos_weight=5. LightGBM: seed 42, scale_pos_weight=5.
Best config: drop relay feats, drop hard positives. No CV. python blend_c.py TRAIN TEST OUT"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from features_v3 import build_features_v3
T = 4; RELAY = ["relay_match6h", "relay_amount_matching"]
tr_dir, te_dir, out = sys.argv[1:4]
tr = build_features_v3(tr_dir).merge(pd.read_csv(Path(tr_dir)/"labels.csv")[["TRANSACTION_HASH","is_suspicious","difficulty"]], on="TRANSACTION_HASH", validate="one_to_one")
te = build_features_v3(te_dir)
tr = tr.loc[:, ~tr.columns.duplicated()]; te = te.loc[:, ~te.columns.duplicated()]
cols = [c for c in te.columns if c != "TRANSACTION_HASH" and c not in RELAY and c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])]
tr = tr[~((tr.difficulty == "hard") & (tr.is_suspicious == 1))]
X, y, Xt = tr[cols], tr.is_suspicious.astype(int).to_numpy(), te[cols]
pr = lambda p: rankdata(p)/len(p)
cats = [CatBoostClassifier(iterations=600, depth=7, learning_rate=.04, l2_leaf_reg=6, scale_pos_weight=5, random_seed=s, verbose=False, thread_count=T, allow_writing_files=False).fit(X, y).predict_proba(Xt)[:, 1] for s in (42, 7, 99)]
lgb = LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=31, min_child_samples=30, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=5, scale_pos_weight=5, n_jobs=T, random_state=42, verbose=-1).fit(X, y).predict_proba(Xt)[:, 1]
rc = pr(np.mean([pr(p) for p in cats], 0))
r = .75*rc + .25*pr(lgb) + 1e-3*np.mean(cats + [lgb], 0)
s = np.clip((r - r.min())/(r.max() - r.min() + 1e-12), 0, 1)
sub = pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": s})
assert len(sub) == 40101 and sub.TRANSACTION_HASH.is_unique and sub.score.between(0, 1).all()
Path(out).parent.mkdir(parents=True, exist_ok=True); sub.to_csv(out, index=False); print("wrote", out, len(cols), "features")
