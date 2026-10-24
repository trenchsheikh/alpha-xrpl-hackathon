"""Post-hoc neighbour boost on OOF scores with nested (leave-fold-out) tuning of alpha,q.
mask='all' : every tx gets max(own, alpha*max score of other txs within +/-W s) if that max is in top q.
mask='self': only ACCOUNT==Destination txs (none exist in this data -> no-op)."""
import sys, json; sys.path[:0] = ["tools", "."]
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as aps
from grouped_cv import metrics
def neigh_max(t, s, W=120):
    o = np.argsort(t, kind="stable"); ts, ss = t[o], s[o]; lo = np.searchsorted(ts, ts - W); hi = np.searchsorted(ts, ts + W, side="right")
    nm = np.zeros(len(s))
    for i, (a, b) in enumerate(zip(lo, hi)):
        seg = np.r_[ss[a:i], ss[i + 1:b]]; nm[i] = seg.max() if len(seg) else 0
    out = np.zeros(len(s)); out[o] = nm; return out
def boost(s, nm, mask, a, q):
    thr = np.quantile(s, 1 - q); m = mask & (nm >= thr); r = s.copy(); r[m] = np.maximum(s[m], a * nm[m]); return r
if __name__ == "__main__":
    oof = np.load(sys.argv[1]); feats = sys.argv[2]
    d = pd.read_parquet(feats, columns=["TRANSACTION_HASH"]).merge(pd.read_csv("data/train/labels.csv"), on="TRANSACTION_HASH")
    tx = pd.read_parquet("data/train/transactions.parquet", columns=["TRANSACTION_HASH", "TIMESTAMP"]); d = d.merge(tx, on="TRANSACTION_HASH")
    from sklearn.model_selection import StratifiedGroupKFold; from grouped_cv import groups
    y = d.is_suspicious.to_numpy(); fid = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d, y, groups(d))): fid[va] = k
    t = d.TIMESTAMP.to_numpy().astype(np.int64)
    # neighbour max computed within each fold's rows only (mimics scoring a test split on its own)
    nm = np.zeros(len(y))
    for k in range(5):
        m = fid == k; nm[m] = neigh_max(t[m], oof[m])
    grid = [(a, q) for a in (.5, .6, .7, .8, .9, .95) for q in (.01, .02, .03, .05)]
    mask = np.ones(len(y), bool); out = oof.copy(); picks = []
    for k in range(5):
        tr = fid != k
        best = max(grid, key=lambda g: aps(y[tr], boost(oof[tr], nm[tr], mask[tr], *g)))
        va = fid == k; out[va] = boost(oof[va], nm[va], mask[va], *best); picks.append(best)
    for name, s in (("no boost", oof), ("time boost (nested)", out)):
        r = metrics(d, s, fid); k_ = r["by_kind"]
        print(f"{name}: AP {r['AP_x100']} P@1% {r['P@1%']} R@1% {r['R@1%']} folds {r['fold_AP_x100']} easyAP {k_['easy']['AP_vs_neg_x100']} medAP {k_['medium']['AP_vs_neg_x100']}")
    print("picked (alpha,q) per fold:", picks)
