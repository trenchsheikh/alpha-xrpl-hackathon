"""SENTINEL V3.1 — economically constrained AML ranking and bubble maps.

Only label-free chronological money-flow graph, balance, account and network
features are used as predictors. Labels are used only to fit a supervised
binary classifier; family, difficulty, cohort and scenario are diagnostic only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

from features_v3 import features_from_edges_v3
from flow_v31 import add_flow_features, csv_graph_data, risk_diffusion
from train_v2 import scenario_groups
from train_v3 import group_split

FOLDS = (42, 137)
NAMES = ("v3_cat", "v3_lgb", "flow_cat", "flow_lgb")
BASE_END = "relay_amount_matching"


def build(dataset_dir):
    raw = csv_graph_data(dataset_dir)
    raw = raw.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)
    f = features_from_edges_v3(raw)
    f, nxt, prv = add_flow_features(raw, f)
    assert f.TRANSACTION_HASH.is_unique
    return f, nxt, prv


def config(name, n_trees, seed):
    if name == "v3_cat":
        return CatBoostClassifier(iterations=n_trees, depth=6,
                                  learning_rate=.045, l2_leaf_reg=9,
                                  random_seed=seed, loss_function="Logloss",
                                  eval_metric="PRAUC", verbose=False,
                                  thread_count=-1)
    if name == "flow_cat":
        return CatBoostClassifier(iterations=n_trees, depth=6,
                                  learning_rate=.035, l2_leaf_reg=10,
                                  random_seed=seed, loss_function="Logloss",
                                  eval_metric="PRAUC", verbose=False,
                                  thread_count=-1, random_strength=.9)
    if name == "v3_lgb":
        return LGBMClassifier(n_estimators=n_trees, objective="binary",
                              metric="average_precision", num_leaves=23,
                              learning_rate=.045, min_child_samples=85,
                              subsample=.85, subsample_freq=1,
                              colsample_bytree=.85, reg_lambda=5.0,
                              reg_alpha=.2, random_state=seed, verbosity=-1,
                              n_jobs=-1)
    if name == "flow_lgb":
        return LGBMClassifier(n_estimators=n_trees, objective="binary",
                              metric="average_precision", num_leaves=13,
                              learning_rate=.04, min_child_samples=60,
                              subsample=.80, subsample_freq=1,
                              colsample_bytree=.78, reg_lambda=8,
                              reg_alpha=.3, random_state=seed,
                              verbosity=-1, n_jobs=-1)
    raise ValueError(name)


def train_models(X, y, train_idx, val_idx, base_columns, seed,
                 tree_counts=None):
    preds, models, rounds = {}, {}, {}
    for name in NAMES:
        xx = X[base_columns] if name.startswith("v3_") else X
        limit = int(tree_counts[name]) if tree_counts else 1100
        model = config(name, limit, seed)
        if name.endswith("cat"):
            validation_args = dict(eval_set=(xx.iloc[val_idx],y.iloc[val_idx]),
                                   early_stopping_rounds=75) if val_idx is not None else {}
            model.fit(xx.iloc[train_idx], y.iloc[train_idx], **validation_args)
            best = int(model.best_iteration_) + 1 if val_idx is not None else limit
        else:
            validation_args = dict(
                eval_set=[(xx.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[early_stopping(80, verbose=False)]
            ) if val_idx is not None else {}
            model.fit(xx.iloc[train_idx], y.iloc[train_idx], **validation_args)
            best = int(model.best_iteration_) if val_idx is not None else limit
        best = max(100, min(limit, best))
        rounds[name] = best
        models[name] = model
        if val_idx is not None:
            preds[name] = model.predict_proba(xx.iloc[val_idx])[:,1]
            score = average_precision_score(y.iloc[val_idx], preds[name])
            print(f"   {name} grouped AP × 100 = {100*score:.2f}, best trees {best}", flush=True)
        else:
            print(f"   {name} full fit {best} trees", flush=True)
    return models, preds, rounds


def possible_scores(outputs, graph_nxt=None, graph_prv=None,
                    valid_indices=None, total_n=None):
    """Predeclared, tested ranking candidates. No label-based propagation."""
    p3 = .5 * outputs["v3_cat"] + .5 * outputs["v3_lgb"]
    flow = .5 * outputs["flow_cat"] + .5 * outputs["flow_lgb"]
    pred = {
        "v3_base": p3,
        "flow_cat": outputs["flow_cat"],
        "flow_lgb": outputs["flow_lgb"],
        "flow_ensemble": flow,
        "mix_25": .75 * p3 + .25 * flow,
        "mix_50": .50 * p3 + .50 * flow,
        "mix_75": .25 * p3 + .75 * flow,
        "rank_mix_50": .5 * rank(p3) + .5 * rank(flow),
        "rank_mix_75": .25 * rank(p3) + .75 * rank(flow),
    }
    if graph_nxt is not None and graph_prv is not None:
        assert total_n is not None
        seeds = np.zeros(total_n, dtype=float)
        vi = np.arange(total_n) if valid_indices is None else valid_indices
        # Seed ONLY held-out predictions during grouped validation.
        # Training-label-dependent in-sample predictions may NOT influence
        # validation transaction risks through graph propagation.
        seeds[vi] = .5 * p3 + .5 * flow
        support = risk_diffusion(seeds, graph_nxt, graph_prv)
        # Risk is not automatically assigned to all neighbours; each link's
        # value/timing and wallet degree affect limited support.
        support = support[vi]
        for name in ("mix_50", "mix_75", "flow_ensemble"):
            raw = rank(pred[name])
            for factor in (.04, .12):
                pred[f"{name}_trace_{factor:g}"] = raw + factor * support
    return pred


def rank(x):
    return rankdata(np.asarray(x), method="average") / (len(x) + 1)


def difficulty_metrics(difficulty, y, prediction, fraction=.01):
    p = np.asarray(prediction)
    yy = np.asarray(y)
    dd = np.asarray(difficulty).astype(str)
    top = np.argsort(-p)[:max(1, int(np.ceil(len(p)*fraction)))]
    table = {}
    for level in ("easy", "medium", "hard"):
        total = int(((dd == level) & (yy == 1)).sum())
        hits = int(((dd[top] == level) & (yy[top] == 1)).sum())
        table[level] = {"found":hits,"total":total,"recall":hits / max(1,total)}
    # This AP measures hard-labelled positives against ALL benign transactions.
    # It is a diagnostic, not the competition's overall AP.
    hard = ((dd == "hard") & (yy == 1))
    eligible = hard | (yy == 0)
    hard_ap = float(average_precision_score(hard[eligible], p[eligible]))
    return table, hard_ap


def main():
    parser = argparse.ArgumentParser(description="SENTINEL economic-flow V3.1")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions_v31.csv")
    parser.add_argument("--folds", type=int, choices=(1,2), default=2)
    parser.add_argument("--no-map", action="store_true")
    args = parser.parse_args()

    print("[1/5] Build V3.1 transaction-level graph, subset-flow and wallet features",
          flush=True)
    features, graph_nxt, graph_prv = build(args.train)
    data = features.merge(pd.read_csv(Path(args.train)/"labels.csv"),
                          on="TRANSACTION_HASH", how="left", validate="one_to_one")
    assert len(features) == len(data)
    assert data.is_suspicious.notna().all()
    y = data.is_suspicious.astype(int).reset_index(drop=True)
    X = features.drop(columns=["TRANSACTION_HASH"]).astype("float32")
    base_index = list(X.columns).index(BASE_END)+1
    base_columns = list(X.columns[:base_index])
    assert len(base_columns) >= 180
    print(f"Features {X.shape[1]} = V3 {len(base_columns)} + "
          f"economic flow {X.shape[1]-len(base_columns)}", flush=True)
    groups = scenario_groups(data)
    all_aps = []
    round_lists = []
    reports = []
    for fold, seed in enumerate(FOLDS[:args.folds], start=1):
        tr, va = group_split(y, groups, seed)
        print(f"[2/5] Fold {fold}/{args.folds}, {len(va)} holdout rows, "
              f"{int(y.iloc[va].sum())} positives", flush=True)
        _, vpred, iterations = train_models(X,y,tr,va,base_columns,seed)
        scores = possible_scores(vpred,graph_nxt,graph_prv,va,len(X))
        metrics = {}
        for name, score in scores.items():
            ap = float(average_precision_score(y.iloc[va],score))
            recall, hard_ap = difficulty_metrics(
                data.iloc[va].difficulty.to_numpy(),
                y.iloc[va].to_numpy(), score)
            metrics[name] = {"ap":ap, "hard_ap":hard_ap, "top1":recall}
        summary = sorted(metrics, key=lambda name: -metrics[name]["ap"])
        for name in summary[:7]:
            m = metrics[name]
            print(f"   {name:29s} AP {100*m['ap']:6.2f} "
                  f"hard-vs-benign AP {100*m['hard_ap']:5.2f} "
                  f"top1 easy/med/hard "
                  f"{m['top1']['easy']['found']}/"
                  f"{m['top1']['medium']['found']}/"
                  f"{m['top1']['hard']['found']}",flush=True)
        print("   V3 baseline:", round(100*metrics["v3_base"]["ap"],2),flush=True)
        all_aps.append({k:v["ap"] for k,v in metrics.items()})
        reports.append({"fold":fold,"seed":seed,"stats":metrics})
        round_lists.append(iterations)
    average = {key:float(np.mean([item[key] for item in all_aps]))
               for key in all_aps[0]}
    print("[3/5] Mean scenario-grouped scores:",flush=True)
    for name, val in sorted(average.items(),key=lambda x:-x[1]):
        print(f"   {name:29s} {val*100:.2f}",flush=True)
    selected = max(average,key=average.get)
    # Gate overly complex model/propagation behind meaningful grouped-AP uplift.
    if average[selected] < average["v3_base"]+.003:
        selected = "v3_base"
    print("SELECTED:",selected,"grouped AP",
          round(100*average[selected],2),flush=True)

    best_trees = {k:int(np.median([r[k] for r in round_lists]))
                  for k in NAMES}
    print("[4/5] Fit full training set",flush=True)
    fit,_,_=train_models(X,y,np.arange(len(X)),None,base_columns,
                         seed=42,tree_counts=best_trees)
    del features, data

    print("[5/5] Build test graph, rank and produce predictions",flush=True)
    test, test_nxt, test_prv = build(args.test)
    ids = test.TRANSACTION_HASH.copy()
    assert ids.is_unique
    TX = test.drop(columns=["TRANSACTION_HASH"]).astype("float32")
    assert list(TX.columns) == list(X.columns), "Train/test columns differ"
    predictions={}
    for name in NAMES:
        dataX=TX[base_columns] if name.startswith("v3_") else TX
        predictions[name] = fit[name].predict_proba(dataX)[:,1]
    score = possible_scores(predictions,test_nxt,test_prv,
                            total_n=len(TX))[selected]
    score = rank(score)  # competition AP depends on ranking, not thresholds
    result = pd.DataFrame({"TRANSACTION_HASH":ids,"score":score})
    assert len(result)==40101 and result.score.notna().all()
    assert result.TRANSACTION_HASH.is_unique and result.score.between(0,1).all()
    out=Path(args.output)
    result.to_csv(out,index=False)
    out.with_suffix(".report.json").write_text(json.dumps({
        "selected":selected,"folds":reports,"mean_group_ap":average,
        "model_iterations":best_trees,"train_size":len(X),
        "test_size":len(TX),"n_features":len(X.columns),
        "existing_v2_official_score":52.5,
        "no_label_based_graph_seeds":True,
        "risk_propagation": "soft, confidence-capped, temporal/value flow only"
    },indent=2))
    print("DONE",str(out),len(result),"selected",selected,
          "grouped AP",round(100*average[selected],2),flush=True)
    if not args.no_map:
        try:
            from bubble_map_v31 import create_bubble_map
            create_bubble_map(args.test,out,"sentinel_v31_bubble_map.html")
        except Exception as exc:
            print("Map generation failed without affecting valid predictions:",
                  repr(exc),flush=True)


if __name__ == "__main__":
    main()
