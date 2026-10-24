"""Supervised scenario linking: pairwise same-scenario model -> per-tx component features.
Train side is out-of-fold on the grouped_cv folds (StratifiedGroupKFold(5, shuffle, 42), cohort>scenario>tx groups)."""
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

WIN, HUB = 7200, 400
MOT = None

def _pairs(d):
    """Candidate pairs (i<j by time) sharing a wallet within WIN, wallet degree<=HUB."""
    N = len(d); ts = d.ts.values
    w = np.concatenate([d.src.values, d.dst.values]); ix = np.concatenate([np.arange(N), np.arange(N)])
    role = np.concatenate([np.zeros(N, int), np.ones(N, int)])  # 0 = wallet is sender, 1 = receiver
    deg = pd.Series(w).map(pd.Series(w).value_counts()).values
    I, J, RI, RJ, D = [], [], [], [], []
    for g in pd.Series(np.arange(2 * N)).groupby(w).indices.values():
        if len(g) < 2 or deg[g[0]] > HUB: continue
        g = g[np.argsort(ts[ix[g]], kind="stable")]; tt = ts[ix[g]]
        hi = np.searchsorted(tt, tt + WIN, "right")
        for a in range(len(g)):
            b = np.arange(a + 1, hi[a])
            if len(b) == 0: continue
            I.append(np.full(len(b), ix[g[a]])); J.append(ix[g[b]]); RI.append(np.full(len(b), role[g[a]])); RJ.append(role[g[b]])
            D.append(np.full(len(b), deg[g[a]]))
    P = pd.DataFrame(dict(i=np.concatenate(I), j=np.concatenate(J), ri=np.concatenate(RI), rj=np.concatenate(RJ), wdeg=np.concatenate(D)))
    P = P[P.i != P.j]
    sw = P.i > P.j; P.loc[sw, ["i", "j"]] = P.loc[sw, ["j", "i"]].values; P.loc[sw, ["ri", "rj"]] = P.loc[sw, ["rj", "ri"]].values
    P["code"] = P.ri * 2 + P.rj  # 0 ss,1 s->r... (earlier sender / later receiver), 2 r->s, 3 rr
    P = P.groupby(["i", "j"], as_index=False, sort=True).agg(code=("code", "min"), wdeg=("wdeg", "min"))
    return P

def _pair_feats(d, P, M):
    i, j = P.i.values, P.j.values; ts = d.ts.values; u = d.u.values + 1e-6
    return pd.DataFrame(dict(dt=ts[j] - ts[i], code=P.code.values, lur=np.abs(np.log(u[j] / u[i])),
                             same_asset=(d.asset.values[i] == d.asset.values[j]).astype(int), wdeg=P.wdeg.values,
                             first=np.maximum(np.maximum(d.sfirst.values[j], d.rfirst.values[j]), 0)))

def _prep(edges):
    d = pd.DataFrame(dict(ts=edges.TIMESTAMP.values.astype(float), src=edges.FROM_ADDRESS.astype(str).values,
                          dst=edges.TO_ADDRESS.astype(str).values, u=edges.VALUE.values.astype(float),
                          asset=edges.ASSET.astype(str).values if "ASSET" in edges else "na"))
    w = pd.concat([d.src, d.dst]); vc = w.value_counts()
    d["sdeg"] = d.src.map(vc).values; d["rdeg"] = d.dst.map(vc).values
    # first-seen: is this the first tx touching that wallet
    order = np.arange(len(d)); allw = pd.DataFrame(dict(w=w.values, o=np.concatenate([order, order]))).groupby("w").o.min()
    d["sfirst"] = (d.src.map(allw).values == order).astype(int); d["rfirst"] = (d.dst.map(allw).values == order).astype(int)
    return d

def _model():
    return LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=63, subsample=.8, subsample_freq=1,
                          colsample_bytree=.8, min_child_samples=40, n_jobs=1, verbose=-1)

def _tx_feats(d, P, p, M):
    N = len(d); out = {}
    out["lk_sum"] = np.bincount(P.i, p, N) + np.bincount(P.j, p, N)
    mx = np.zeros(N); np.maximum.at(mx, P.i.values, p); np.maximum.at(mx, P.j.values, p); out["lk_max"] = mx
    k = p > 0.5; A = coo_matrix((np.ones(k.sum()), (P.i.values[k], P.j.values[k])), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    out["lk_cn5"] = np.bincount(lab)[lab]
    Mg = M.groupby(lab)
    for c, v in Mg.max().reindex(lab).items(): out[f"lk_{c}_cmax5"] = v.values
    for c, v in Mg.mean().reindex(lab).items(): out[f"lk_{c}_cmean5"] = v.values
    return pd.DataFrame(out).fillna(0)

def _motif(feat):
    cols = [c for c in feat.columns if c.startswith("motif_") and "_w" not in c and "_h2" not in c]
    return feat[cols].reset_index(drop=True).astype(float)

def _load_train(train_dir):
    from payments import load_payments
    from features_v3 import features_from_edges_v3
    raw = pd.DataFrame(load_payments(train_dir)).sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)
    return raw, features_from_edges_v3(raw)

def _labels(hashes, train_dir):
    lab = pd.read_csv(Path(train_dir) / "labels.csv")
    L = pd.DataFrame(dict(TRANSACTION_HASH=hashes)).merge(lab, on="TRANSACTION_HASH", how="left", validate="one_to_one")
    grp = np.where(L.cohort_id.notna(), "c:" + L.cohort_id.astype(str),
          np.where(L.scenario_id.notna(), "s:" + L.scenario_id.astype(str), "t:" + L.TRANSACTION_HASH))
    return L, grp

def link_features(raw, feat, dataset_dir, return_debug=False):
    """raw/feat aligned (sorted by TIMESTAMP,HASH). Returns DataFrame of lk_* columns."""
    dataset_dir = Path(dataset_dir)
    is_train = (dataset_dir / "labels.csv").exists() and dataset_dir.name == "train"
    d = _prep(raw); M = _motif(feat); P = _pairs(d); X = _pair_feats(d, P, M)
    if is_train:
        L, grp = _labels(raw.TRANSACTION_HASH.values, dataset_dir)
        sid = L.scenario_id.fillna("").values
        y = ((sid[P.i] == sid[P.j]) & (sid[P.i] != "")).astype(int)
        # grouped_cv fold ids (same split call as tools/grouped_cv.py: f.merge(lab) order == raw order)
        fold = np.zeros(len(raw), int)
        for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(raw)), L.is_suspicious.values.astype(int), grp)):
            fold[va] = k
        fi, fj = fold[P.i.values], fold[P.j.values]
        p = np.zeros(len(P))
        for k in range(5):
            tr = (fi != k) & (fj != k); te = (fi == k) | (fj == k)
            m = _model().fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
        if return_debug: return _tx_feats(d, P, p, M), P, X, y, p, fold, L
    else:
        tdir = dataset_dir.parent / "train"
        traw, tfeat = _load_train(tdir)
        td = _prep(traw); tP = _pairs(td); tX = _pair_feats(td, tP, _motif(tfeat))
        L, _ = _labels(traw.TRANSACTION_HASH.values, tdir); sid = L.scenario_id.fillna("").values
        ty = ((sid[tP.i] == sid[tP.j]) & (sid[tP.i] != "")).astype(int)
        p = _model().fit(tX, ty).predict_proba(X)[:, 1]
    return _tx_feats(d, P, p, M)
