"""Grouped 5-fold CV for ensemble_boost configs on a cached feature parquet.
Usage: python tools/ensemble_cv.py --feats cache/X.parquet --exp NAME [--exp NAME ...]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_boost as eb
from grouped_cv import groups, metrics, META

def member_oof(d, X, members, cfg, seed=42):
    """OOF for each (kind,seed) member, cached per member."""
    y = d.is_suspicious.to_numpy().astype(int); out = {}
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups(d)))
    for kind, s in members:
        key = f"{kind}{s}_{cfg['tag']}"; p = Path(f"out/ens/{key}.npy")
        if p.exists(): out[(kind, s)] = np.load(p); continue
        oof = np.zeros(len(y))
        for tr, va in folds:
            m = eb.make(kind, s, cfg); m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        p.parent.mkdir(parents=True, exist_ok=True); np.save(p, oof); out[(kind, s)] = oof
        print(f"  member {key}: AP {100*aps(y, oof):.2f}", flush=True)
    fid = np.zeros(len(y), int)
    for k, (_, va) in enumerate(folds): fid[va] = k
    return out, fid

def rankavg(ps, w=None):
    w = w or [1] * len(ps); return sum(wi * rankdata(p) / len(p) for wi, p in zip(w, ps)) / sum(w)
