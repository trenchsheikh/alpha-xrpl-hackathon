"""b_ batch: branch features_v3.build_features_v3 -> 0.75 CatBoost s42 / 0.25 LightGBM s42 percentile-rank blend (best config: drop relay feats, drop hard positives). No CV.
python blend_b.py TRAIN_DIR TEST_DIR OUT_CSV"""
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
cols = [c for c in te.columns if c != "TRANSACTION_HASH" and c not in RELAY and c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])]
tr = tr[~((tr.difficulty == "hard") & (tr.is_suspicious == 1))]
X, y, Xt = tr[cols], tr.is_suspicious.astype(int).to_numpy(), te[cols]
cat = CatBoostClassifier(iterations=600, depth=7, learning_rate=.04, l2_leaf_reg=6, random_seed=42, verbose=False, thread_count=T, allow_writing_files=False)
lgb = LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=31, min_child_samples=30, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=5, n_jobs=T, random_state=42, verbose=-1)
ps = [m.fit(X, y).predict_proba(Xt)[:, 1] for m in (cat, lgb)]
r = .75*rankdata(ps[0])/len(ps[0]) + .25*rankdata(ps[1])/len(ps[1]) + 1e-3*np.mean(ps, 0)
s = np.clip((r - r.min())/(r.max() - r.min() + 1e-12), 0, 1)
sub = pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": s})
assert len(sub) == 40101 and sub.TRANSACTION_HASH.is_unique and sub.score.between(0, 1).all()
Path(out).parent.mkdir(parents=True, exist_ok=True); sub.to_csv(out, index=False); print("wrote", out, len(cols), "features")
