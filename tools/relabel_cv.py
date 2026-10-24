import sys; sys.path[:0] = ["tools", "."]
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from grouped_cv import groups, metrics, META
import ensemble_boost as eb
f = pd.read_parquet(sys.argv[1]); d = f.merge(pd.read_csv("data/train/labels.csv"), on="TRANSACTION_HASH")
drop = [c for c in d if "relay" in c]
X = d.drop(columns=["TRANSACTION_HASH"] + META + drop).select_dtypes("number"); y = d.is_suspicious.to_numpy()
hard = (d.difficulty == "hard").to_numpy() & (y == 1)
for mode in sys.argv[2:]:
    oof = np.zeros(len(y)); fid = np.zeros(len(y), int)
    for i, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups(d))):
        fid[va] = i; yt = y.copy()
        if mode == "relabel0": yt[hard] = 0
        if mode == "drophard": tr = tr[~hard[tr]]
        m = eb.make("cat", 42, eb.DEFAULT); m.fit(X.iloc[tr], yt[tr]); oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    np.save(f"out/ens/cat42_norelay_{mode}.npy", oof); r = metrics(d, oof, fid); k = r["by_kind"]
    print(f"{mode}: AP {r['AP_x100']} P@1% {r['P@1%']} R@1% {r['R@1%']} folds {r['fold_AP_x100']} | easy AP {k['easy']['AP_vs_neg_x100']} R5 {k['easy']['recall@5%rank']} R1 {k['easy']['recall@1%']} | med AP {k['medium']['AP_vs_neg_x100']} R5 {k['medium']['recall@5%rank']} R1 {k['medium']['recall@1%']}", flush=True)
