"""Reusable grouped-CV holdout harness for the XRPL Decode & Detect task.

Modes
  1) Score an existing out-of-fold predictions CSV on train:
       python tools/grouped_cv.py --labels DATA/train/labels.csv --oof oof.csv
  2) Run grouped K-fold CV on a feature builder from ANY checkout/branch:
       python tools/grouped_cv.py --repo /path/to/checkout --features MODULE:FUNC \
              --train DATA/train [--folds 5] [--threads 2] [--iters 600]
     FUNC(train_dir) must return a DataFrame with TRANSACTION_HASH + numeric
     features (a tuple whose first item is that DataFrame is also accepted).
     Features are cached under --cache keyed by repo HEAD + module:func.
Groups: cohort_id if set, else scenario_id, else the transaction (same as v2).
Folds: StratifiedGroupKFold(shuffle, seed). Metric = sklearn AP (x100) on pooled OOF,
plus per-fold AP, P/R at top 1% (ties by hash asc), recall@top1% and kind-vs-negatives
AP by difficulty kind (easy/medium/obs_hard/ambiguous) and recall@1% by family."""
import argparse, importlib, json, os, subprocess, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold

META = ["is_suspicious", "family", "difficulty", "scenario_id", "cohort_id"]

def groups(d):
    return np.where(d.cohort_id.notna(), "c:" + d.cohort_id.astype(str),
           np.where(d.scenario_id.notna(), "s:" + d.scenario_id.astype(str), "t:" + d.TRANSACTION_HASH))

def kinds(d):
    k = np.where(d.difficulty == "hard", "obs_hard", d.difficulty.astype(str))
    k = np.where(d.family == "ambiguous", "ambiguous", k)
    return np.where(d.is_suspicious == 1, k, "neg")

def metrics(d, score, folds=None):
    y = d.is_suspicious.to_numpy().astype(int); h = d.TRANSACTION_HASH.to_numpy(); K = kinds(d)
    n = int(np.ceil(len(y) * .01)); top = np.lexsort((h, -score))[:n]
    sel = np.zeros(len(y), bool); sel[top] = True
    r = {"AP_x100": round(100 * aps(y, score), 2), "top1_n": n,
         "P@1%": round(y[sel].mean(), 4), "R@1%": round(y[sel].sum() / y.sum(), 4), "by_kind": {}, "by_family": {}}
    if folds is not None:
        r["fold_AP_x100"] = [round(100 * aps(y[folds == f], score[folds == f]), 2) for f in np.unique(folds)]
    for k in ["easy", "medium", "obs_hard", "ambiguous"]:
        m = K == k; mm = m | (K == "neg")
        r["by_kind"][k] = {"pos": int(m.sum()), "recall@1%": round(sel[m].mean(), 3),
                           "AP_vs_neg_x100": round(100 * aps(y[mm], score[mm]), 2),
                           "recall@5%rank": round((np.argsort(np.argsort(-score))[m] < 5 * n).mean(), 3)}
    for f in sorted(set(d.family[y == 1])):
        m = (d.family.to_numpy() == f) & (y == 1)
        r["by_family"][f] = {"pos": int(m.sum()), "recall@1%": round(sel[m].mean(), 3)}
    return r

def show(r):
    print(f"AP x100 = {r['AP_x100']}  P@1% = {r['P@1%']}  R@1% = {r['R@1%']}  (top1 n={r['top1_n']})")
    if "fold_AP_x100" in r: print("fold AP:", r["fold_AP_x100"])
    for k, v in r["by_kind"].items(): print(f"  {k:9s} {v}")
    for k, v in r["by_family"].items(): print(f"    {k:18s} {v}")

def build_features(repo, spec, train, cache):
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "nogit"
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
    key = hashlib.sha1(f"{Path(repo).resolve()}|{head}|{dirty}|{spec}|{Path(train).resolve()}".encode()).hexdigest()[:12]
    c = Path(cache) / f"feats_{key}.parquet"
    if c.exists(): print("cached features", c); return pd.read_parquet(c)
    sys.path.insert(0, str(Path(repo).resolve())); cwd = os.getcwd(); os.chdir(repo)
    try:
        mod, fn = spec.split(":"); out = getattr(importlib.import_module(mod), fn)(str(Path(cwd, train).resolve()))
    finally: os.chdir(cwd)
    f = out[0] if isinstance(out, tuple) else out
    c.parent.mkdir(parents=True, exist_ok=True); f.to_parquet(c); print("wrote", c); return f

def cv(d, X, nfold, seed, iters, threads, model):
    y = d.is_suspicious.to_numpy().astype(int); oof = np.zeros(len(y)); fid = np.zeros(len(y), int)
    for k, (tr, va) in enumerate(StratifiedGroupKFold(nfold, shuffle=True, random_state=seed).split(X, y, groups(d))):
        if model == "lgb":
            from lightgbm import LGBMClassifier
            m = LGBMClassifier(n_estimators=iters, learning_rate=.04, num_leaves=63, subsample=.8, subsample_freq=1,
                               colsample_bytree=.7, n_jobs=threads, random_state=seed, verbose=-1)
        else:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(iterations=iters, depth=7, learning_rate=.04, l2_leaf_reg=6, random_seed=seed,
                                   verbose=False, thread_count=threads, allow_writing_files=False)
        m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict_proba(X.iloc[va])[:, 1]; fid[va] = k
        print(f" fold {k}: AP x100 {100*aps(y[va], oof[va]):.2f} ({y[va].sum()} pos)", flush=True)
    return oof, fid

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--labels", default="data/train/labels.csv"); a.add_argument("--train", default="data/train")
    a.add_argument("--oof"); a.add_argument("--repo", default="."); a.add_argument("--features")
    a.add_argument("--drop", default="", help="comma list of feature columns to drop")
    a.add_argument("--folds", type=int, default=5); a.add_argument("--seed", type=int, default=42)
    a.add_argument("--iters", type=int, default=600); a.add_argument("--threads", type=int, default=2)
    a.add_argument("--model", choices=["cat", "lgb"], default="cat")
    a.add_argument("--cache", default="cache"); a.add_argument("--tag", default="run"); a.add_argument("--outdir", default="out/cv")
    a = a.parse_args()
    lab = pd.read_csv(a.labels); Path(a.outdir).mkdir(parents=True, exist_ok=True)
    if a.oof:
        p = pd.read_csv(a.oof); d = lab.merge(p, on="TRANSACTION_HASH", validate="one_to_one")
        assert len(d) == len(lab), f"oof covers {len(d)}/{len(lab)} train rows"
        r = metrics(d, d.score.to_numpy())
    else:
        f = build_features(a.repo, a.features, a.train, a.cache)
        d = f.merge(lab, on="TRANSACTION_HASH", validate="one_to_one"); assert len(d) == len(lab)
        X = d.drop(columns=["TRANSACTION_HASH"] + META + [c for c in a.drop.split(",") if c]).select_dtypes("number")
        print(f"{len(X.columns)} features, {len(d)} rows, {d.is_suspicious.sum()} pos", flush=True)
        oof, fid = cv(d, X, a.folds, a.seed, a.iters, a.threads, a.model)
        pd.DataFrame({"TRANSACTION_HASH": d.TRANSACTION_HASH, "score": oof, "fold": fid}).to_csv(f"{a.outdir}/{a.tag}_oof.csv", index=False)
        r = metrics(d, oof, fid); r.update(features=len(X.columns), folds=a.folds, iters=a.iters, model=a.model, spec=a.features, repo=a.repo)
    show(r); Path(f"{a.outdir}/{a.tag}_metrics.json").write_text(json.dumps(r, indent=2)); print("saved", f"{a.outdir}/{a.tag}_metrics.json")
