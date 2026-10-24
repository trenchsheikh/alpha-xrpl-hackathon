import sys, json; from pathlib import Path
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as aps
from ensemble_cv import member_oof, rankavg
import ensemble_boost as eb
from grouped_cv import metrics, META
feats = sys.argv[1]; stage = sys.argv[2]
f = pd.read_parquet(feats); lab = pd.read_csv("data/train/labels.csv")
d = f.merge(lab, on="TRANSACTION_HASH", validate="one_to_one")
X0 = d.drop(columns=["TRANSACTION_HASH"] + META).select_dtypes("number"); y = d.is_suspicious.to_numpy()
res = {}
def rep(name, s, fid):
    r = metrics(d, s, fid); res[name] = r
    print(f"{name:28s} AP {r['AP_x100']:6.2f} folds {r['fold_AP_x100']} med R@1% {r['by_kind']['medium']['recall@1%']} medAP {r['by_kind']['medium']['AP_vs_neg_x100']} easyR {r['by_kind']['easy']['recall@1%']}", flush=True)
cfg = dict(eb.DEFAULT)
def drops(name):
    if name == "full": return []
    if name == "noclock": return ["hour", "time_fraction"] + [c for c in X0 if "hour" in c.lower()]
    if name == "norelay": return [c for c in X0 if "relay" in c or "cohort" in c]
    if name.startswith("imp"):
        # importance pruning from a CatBoost fit inside each training fold is costly; use one fit on fold-0 train only
        return json.load(open(f"out/ens/{name}.json"))
    raise ValueError(name)
if stage == "members":
    mem = [("cat", 42), ("cat", 7), ("cat", 99), ("lgb", 42), ("lgb", 7), ("lgb", 99), ("xgb", 42), ("xgb", 7)]
    cfg["tag"] = "full"; o, fid = member_oof(d, X0, mem, cfg)
    for k, v in o.items(): rep(f"{k[0]}{k[1]}", v, fid)
    C = [o[k] for k in mem if k[0] == "cat"]; L = [o[k] for k in mem if k[0] == "lgb"]; Xg = [o[k] for k in mem if k[0] == "xgb"]
    rep("cat x3", rankavg(C), fid); rep("lgb x3", rankavg(L), fid); rep("xgb x2", rankavg(Xg), fid)
    rep("cat3+lgb3", rankavg(C + L), fid); rep("cat3+xgb2", rankavg(C + Xg), fid); rep("all8", rankavg(C + L + Xg), fid)
    for w in (.25, .5, .75):
        rep(f"cat{w}/lgb rankmix", w * rankavg(C) + (1 - w) * rankavg(L), fid)
        rep(f"cat{w}/(lgb+xgb)", w * rankavg(C) + (1 - w) * rankavg(L + Xg), fid)
if stage == "prune":
    for name in sys.argv[3:]:
        dr = drops(name); X = X0.drop(columns=[c for c in dr if c in X0]); cfg["tag"] = name
        o, fid = member_oof(d, X, [("cat", 42), ("lgb", 42)], cfg)
        rep(f"{name}({len(dr)}) cat42", o[("cat", 42)], fid); rep(f"{name} lgb42", o[("lgb", 42)], fid)
        rep(f"{name} cat+lgb", rankavg([o[("cat", 42)], o[("lgb", 42)]]), fid)
Path(f"out/ens/res_{stage}_{'_'.join(sys.argv[3:])}.json").write_text(json.dumps(res, indent=1, default=float))
