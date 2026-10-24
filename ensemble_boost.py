"""dev-boost: rank-average ensemble (CatBoost / LightGBM / XGBoost, multi-seed) + feature pruning.
Works on ANY feature DataFrame (TRANSACTION_HASH + numeric columns), so it stacks on any feature branch."""
import numpy as np, pandas as pd
from scipy.stats import rankdata

THREADS = 3
DEFAULT = {"members": [("cat", 42), ("cat", 7), ("lgb", 42), ("lgb", 7)], "weights": None,
           "cat_iters": 600, "lgb_iters": 400, "xgb_iters": 400, "drop": []}

def make(kind, seed, cfg):
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=cfg["cat_iters"], depth=7, learning_rate=.04, l2_leaf_reg=6,
                                  random_seed=seed, verbose=False, thread_count=THREADS, allow_writing_files=False)
    if kind == "lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=cfg["lgb_iters"], learning_rate=.03, num_leaves=31, min_child_samples=30,
                              subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=5,
                              n_jobs=THREADS, random_state=seed, verbose=-1)
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=cfg["xgb_iters"], learning_rate=.04, max_depth=6, subsample=.8,
                             colsample_bytree=.5, min_child_weight=3, reg_lambda=5, n_jobs=THREADS,
                             random_state=seed, tree_method="hist", eval_metric="aucpr")
    raise ValueError(kind)

def prune(X, cfg):
    return X.drop(columns=[c for c in cfg.get("drop", []) if c in X.columns])

def fit_predict(Xtr, ytr, Xte, cfg=DEFAULT, return_members=False):
    """Fit every member on (Xtr,ytr); return rank-averaged score in [0,1] for Xte."""
    Xtr, Xte = prune(Xtr, cfg), prune(Xte, cfg)[prune(Xtr, cfg).columns]
    preds = []
    for kind, seed in cfg["members"]:
        m = make(kind, seed, cfg); m.fit(Xtr, ytr); preds.append(m.predict_proba(Xte)[:, 1])
    w = np.asarray(cfg["weights"] or [1.0] * len(preds), float)
    r = sum(wi * rankdata(p) / len(p) for wi, p in zip(w, preds)) / w.sum()
    # tiny probability-mean term keeps ties broken sensibly; final scaled to [0,1]
    s = r + 1e-3 * np.mean(preds, axis=0)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return (s, preds) if return_members else s
