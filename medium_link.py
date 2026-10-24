"""Medium-difficulty linking features (label-free, per-tx; identical for train/val/test)."""
import numpy as np, pandas as pd

def _uf_find(par, x):
    r = x
    while par[r] != r: r = par[r]
    while par[x] != r: par[x], x = r, par[x]
    return r

def comp_feats(m, lo=0.8, hi=1.15, g0=300, g1=2400):
    """Link tx i -> j when dst_i == src_j, USD ratio in [lo,hi], gap in [g0,g1]; score components."""
    N = len(m); ts = m.ts.values.astype(float); u = m.u.values.astype(float)
    s = pd.Series(pd.factorize(np.r_[m.src.values, m.dst.values])[0])
    sc = s.values[:N]; dc = s.values[N:]
    o = np.lexsort((ts, sc)); key_s = sc[o]; ts_s = ts[o]
    BIG = 1e10
    k = key_s * BIG + ts_s
    a = np.searchsorted(k, dc * BIG + ts + g0, 'left'); b = np.searchsorted(k, dc * BIG + ts + g1, 'right')
    cnt = b - a
    I = np.repeat(np.arange(N), cnt)
    J = o[np.concatenate([np.arange(x, y) for x, y in zip(a, b)])] if cnt.sum() else np.array([], int)
    ok = (sc[J] != sc[I]) & (u[J] / np.maximum(u[I], 1e-9) >= lo) & (u[J] / np.maximum(u[I], 1e-9) <= hi) if len(J) else np.array([], bool)
    I, J = I[ok], J[ok]
    par = list(range(N))
    for i, j in zip(I.tolist(), J.tolist()):
        ri, rj = _uf_find(par, i), _uf_find(par, j)
        if ri != rj: par[ri] = rj
    root = np.array([_uf_find(par, i) for i in range(N)])
    g = pd.DataFrame(dict(c=root, ts=ts)).sort_values(['c', 'ts'])
    gap = g.ts.diff().where(g.c.eq(g.c.shift()))
    gg = gap.groupby(g.c)
    size = g.groupby('c').size(); dur = g.groupby('c').ts.agg(lambda x: x.max() - x.min()) if False else g.groupby('c').ts.max() - g.groupby('c').ts.min()
    gm, gs, gn = gg.mean(), gg.std(ddof=0), gg.count()
    cv = (gs / gm).where((gn > 1) & (gm > 0), 9.0).fillna(9.0)
    sz = size.reindex(root).values.astype(float); du = dur.reindex(root).values / 3600; cvv = cv.reindex(root).values
    score = ((sz >= 3) & (sz <= 16)) * np.exp(-((np.log(np.clip(du, 1e-3, None)) - np.log(1.6)) ** 2) / 0.5) / (1 + cvv)
    return dict(score=score, size=sz, dur=du, gapcv=cvv)

def slow_rhythm(m, g0=600, g1=2000, W=7200):
    N = len(m); ts = m.ts.values.astype(float)
    codes = pd.factorize(np.r_[m.src.values, m.dst.values])[0]; sc, dc = codes[:N], codes[N:]
    w = np.r_[sc, dc]; cp = np.r_[dc, sc]; t = np.r_[ts, ts]
    L = pd.DataFrame(dict(w=w, cp=cp, t=t)).sort_values(['w', 't'], kind='mergesort')
    L['new'] = ~L.duplicated(['w', 'cp'])
    Ln = L[L.new].copy()
    same = Ln.w.eq(Ln.w.shift()); gp = Ln.t.diff().where(same)
    gn = (-Ln.t.diff(-1)).where(Ln.w.eq(Ln.w.shift(-1)))
    Ln['good'] = (gp.between(g0, g1) | gn.between(g0, g1)).astype(float)
    BIG = 1e10; k = Ln.w.values * BIG + Ln.t.values; cs = np.r_[0, np.cumsum(Ln.good.values)]
    def q(c):
        a = np.searchsorted(k, c * BIG + ts - W, 'left'); b = np.searchsorted(k, c * BIG + ts + W, 'right')
        return cs[b] - cs[a]
    s_, r_ = q(sc), q(dc)
    return dict(s=s_, r=r_, tot=s_ + r_)

def medium_link_features(d):
    m = pd.DataFrame(dict(ts=d.TIMESTAMP.values.astype(float), src=d.FROM_ADDRESS.astype(str).values,
                          dst=d.TO_ADDRESS.astype(str).values, u=d.VALUE.values.astype(float)))
    out = {}
    for g0, g1 in ((300, 2400), (600, 3600)):
        for kk, v in comp_feats(m, g0=g0, g1=g1).items(): out[f'mlink_{kk}_{g0}_{g1}'] = v
    for kk, v in slow_rhythm(m).items(): out[f'slowrhythm_{kk}'] = v
    return pd.DataFrame(out)
