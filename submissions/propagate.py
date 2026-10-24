"""Label-free mean-of-component score propagation.

Links transactions sharing a wallet (FROM/TO in transfers.parquet) when consecutive
uses of that wallet are <= W seconds apart, ignoring wallets in > K transactions.
score' = rank + a * mean_rank(component) for components with > 2 txns; rescaled to [0,1].

Usage: python propagate.py --pred in.csv --transfers DATA/test/transfers.parquet \
         --out out.csv [--K 400] [--W 600] [--a 0.3]
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import rankdata

def components(n, E, W, K):
    deg = E.groupby("w").i.nunique()
    e = E[E.w.map(deg) <= K].sort_values(["w", "TIMESTAMP"])
    p = np.arange(n)
    def find(a):
        while p[a] != a:
            p[a] = p[p[a]]; a = p[a]
        return a
    a, t, w = e.i.to_numpy(), e.TIMESTAMP.to_numpy(), e.w.to_numpy()
    for j in range(1, len(a)):
        if w[j] == w[j - 1] and t[j] - t[j - 1] <= W:
            x, y = find(a[j]), find(a[j - 1])
            if x != y: p[x] = y
    return np.array([find(i) for i in range(n)])

def propagate(pred, transfers, K=400, W=600, a=0.3):
    idx = pd.Series(np.arange(len(pred)), index=pred.TRANSACTION_HASH)
    T = transfers[["TRANSACTION_HASH", "TIMESTAMP", "FROM_ADDRESS", "TO_ADDRESS"]].copy()
    T["i"] = T.TRANSACTION_HASH.map(idx); T = T.dropna(subset=["i"]); T["i"] = T.i.astype(int)
    E = pd.concat([T[["i", "TIMESTAMP", "FROM_ADDRESS"]].rename(columns={"FROM_ADDRESS": "w"}),
                   T[["i", "TIMESTAMP", "TO_ADDRESS"]].rename(columns={"TO_ADDRESS": "w"})]).dropna().drop_duplicates(["i", "w"])
    c = components(len(pred), E, W, K)
    r = rankdata(pred.score.to_numpy()) / len(pred)
    g = pd.Series(r).groupby(c); sz = g.transform("size").to_numpy(); mn = g.transform("mean").to_numpy()
    s = r + a * mn * (sz > 2)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return pd.DataFrame({"TRANSACTION_HASH": pred.TRANSACTION_HASH, "score": np.clip(s, 0, 1)}), c, sz

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--transfers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=400); ap.add_argument("--W", type=float, default=600)
    ap.add_argument("--a", type=float, default=0.3)
    x = ap.parse_args()
    out, c, sz = propagate(pd.read_csv(x.pred), pd.read_parquet(x.transfers), x.K, x.W, x.a)
    assert list(out.columns) == ["TRANSACTION_HASH", "score"] and out.TRANSACTION_HASH.is_unique and out.score.between(0, 1).all()
    out.to_csv(x.out, index=False)
    print(f"wrote {x.out} rows={len(out)} comps>2={len(set(c[sz > 2]))} txns_in_comps>2={(sz > 2).sum()}")
