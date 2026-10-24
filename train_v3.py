"""SENTINEL V3 — grouped-validation-selected ensemble for synthetic AML ranking.

Reproducible, offline, full-window only. Scenario/cohort/difficulty annotations
are used EXCLUSIVELY for holdout grouping/diagnostics, never as model inputs.
Only 'is_suspicious' is the supervised learning target.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from features_v3 import build_features_v3
from train_v2 import scenario_groups

SEEDS = (42, 137)
BASE_END = "temporal_first_hop_delay"


def group_split(y, groups, seed):
    cv = GroupShuffleSplit(n_splits=24, test_size=.20, random_state=seed)
    best = None
    best_loss = float("inf")
    for tr, va in cv.split(np.arange(len(y)), y, groups):
        pos = int(y.iloc[va].sum())
        if pos < 100 or pos > 450 or y.iloc[tr].sum() < 600:
            continue
        loss = abs(len(va) / len(y) - .2) + 5 * abs(float(y.iloc[va].mean()) - float(y.mean()))
        if loss < best_loss:
            best = (tr, va)
            best_loss = loss
    if best is None:
        raise ValueError("No stratification-compatible scenario split was found")
    return best


def make_cat(branch, iters=1150, seed=42):
    if branch == "v2":
        return CatBoostClassifier(iterations=iters, depth=7, learning_rate=.04,
                                  l2_leaf_reg=6, random_seed=seed, thread_count=-1,
                                  loss_function="Logloss", eval_metric="PRAUC", verbose=False)
    return CatBoostClassifier(iterations=iters, depth=6, learning_rate=.045,
                              l2_leaf_reg=9, random_seed=seed, thread_count=-1,
                              loss_function="Logloss", eval_metric="PRAUC", verbose=False)


def make_light(branch, iters=1100, seed=42):
    expert = branch == "expert"
    return lgb.LGBMClassifier(
        n_estimators=iters, objective="binary", metric="average_precision",
        num_leaves=15 if expert else 23,
        learning_rate=.045, max_depth=-1,
        min_child_samples=65 if expert else 85,
        colsample_bytree=.85, subsample=.85, subsample_freq=1,
        reg_lambda=7.0 if expert else 5.0,
        reg_alpha=.2, random_state=seed, n_jobs=-1, verbosity=-1)


def rank(p):
    return rankdata(p, method="average") / (len(p) + 1)


def candidates(preds):
    # All candidates are PREDECLARED; they are not fitted to validation labels.
    p = {name: rank(score) for name, score in preds.items()}
    return {
        "v2": preds["v2"],
        "v3_cat": preds["v3_cat"],
        "v3_lgb": preds["v3_lgb"],
        "v3_expert": preds["v3_expert"],
        "raw_cat_lgb": .5 * preds["v3_cat"] + .5 * preds["v3_lgb"],
        "rank_cat_lgb": .5 * p["v3_cat"] + .5 * p["v3_lgb"],
        "rank_three": .40 * p["v3_cat"] + .40 * p["v3_lgb"] + .20 * p["v3_expert"],
        "rank_four": .25 * p["v2"] + .35 * p["v3_cat"] + .30 * p["v3_lgb"] + .10 * p["v3_expert"],
        "rank_relay": .65 * p["v3_cat"] + .35 * p["v3_expert"],
        "rank_v2_v3": .35 * p["v2"] + .35 * p["v3_cat"] + .30 * p["v3_lgb"],
    }


def train_branches(features, labels, base_columns, tr, va=None, seed=42, iterations=None):
    X0 = features[base_columns]
    X = features
    y = labels
    res, pred, fitted_iters = {}, {}, {}
    for branch in ("v2", "v3_cat", "v3_lgb", "v3_expert"):
        matrix = X0 if branch == "v2" else X
        wanted = iterations[branch] if iterations else 1150
        model = (make_cat(branch, iters=wanted, seed=seed)
                 if branch in ("v2", "v3_cat") else
                 make_light(branch, iters=wanted, seed=seed))
        if branch == "v3_expert":
            # Upsample edge-isolated patterns using only observable feature values.
            pair = features["pair_count"].to_numpy()
            weights = np.where(pair <= 2, 2.5, 0.65)
            weight_args = {"sample_weight": weights[tr]}
        else:
            weight_args = {}
        if branch in ("v2", "v3_cat"):
            fit_args = {
                "eval_set": (matrix.iloc[va], y.iloc[va]),
                "early_stopping_rounds": 80,
            } if va is not None else {}
            model.fit(matrix.iloc[tr], y.iloc[tr], **weight_args, **fit_args)
            best = int(model.best_iteration_ + 1) if va is not None else wanted
        else:
            fit_args = {
                "eval_set": [(matrix.iloc[va], y.iloc[va])],
                "callbacks": [lgb.early_stopping(75, first_metric_only=True, verbose=False)],
            } if va is not None else {}
            model.fit(matrix.iloc[tr], y.iloc[tr], **weight_args, **fit_args)
            best = int(model.best_iteration_) if va is not None else wanted
        res[branch] = model
        fitted_iters[branch] = int(max(130, min(wanted, best)))
        if va is not None:
            pred[branch] = model.predict_proba(matrix.iloc[va])[:, 1]
            ap = average_precision_score(y.iloc[va], pred[branch])
            print(f"    {branch}: grouped AP {100*ap:.2f}, trees {fitted_iters[branch]}", flush=True)
        else:
            print(f"    trained {branch}: {wanted} trees", flush=True)
    return res, pred, fitted_iters


def test_predictions(models, features, base_columns):
    return {
        name: models[name].predict_proba(
            features[base_columns] if name == "v2" else features
        )[:, 1]
        for name in models
    }


def main():
    parser = argparse.ArgumentParser(description="SENTINEL V3")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions_v3.csv")
    parser.add_argument("--folds", type=int, default=2,
                        choices=(1, 2), help="1 = fast; 2 = more robust model selection")
    a = parser.parse_args()

    print("[1/5] Creating behavioural, graph, and batch-flow features", flush=True)
    train = build_features_v3(a.train)
    data = train.merge(pd.read_csv(Path(a.train)/"labels.csv"),
                       on="TRANSACTION_HASH", how="left", validate="one_to_one")
    assert len(train) == len(data) and data.is_suspicious.notna().all()
    y = data.is_suspicious.astype(int).reset_index(drop=True)
    assert int(y.sum()) == 1250
    X = train.drop(columns=["TRANSACTION_HASH"]).astype("float32")
    ix = list(X.columns).index(BASE_END) + 1
    base_columns = list(X.columns[:ix])
    groups = scenario_groups(data)
    print(f"Training: {len(y)} rows, {y.sum()} positives, "
          f"{len(base_columns)} V2 and {X.shape[1]-len(base_columns)} new V3 features", flush=True)

    fold_scores, fold_models, diagnostic = [], [], []
    for fold, seed in enumerate(SEEDS[:a.folds], start=1):
        tr, va = group_split(y, groups, seed)
        print(f"[2/5] Grouped fold {fold}/{a.folds}: {len(va)} rows, "
              f"{y.iloc[va].sum()} positives", flush=True)
        models, pred, fitted_iters = train_branches(X, y, base_columns, tr, va, seed)
        scores = candidates(pred)
        ap = {name: float(average_precision_score(y.iloc[va], p))
              for name, p in scores.items()}
        fold_scores.append(ap)
        fold_models.append(fitted_iters)
        winning = max(ap, key=ap.get)
        print("   Winner this fold:", winning,
              f"{ap[winning]*100:.2f}/100", flush=True)
        difficulty = data.iloc[va].difficulty.reset_index(drop=True)
        for candidate in ["v2", "v3_cat", "v3_lgb", "rank_three", winning]:
            pp = scores[candidate]
            top = np.argsort(-pp)[:max(1, int(np.ceil(len(pp)*.01)))]
            d = difficulty.iloc[top]
            yy = y.iloc[va].reset_index(drop=True).iloc[top]
            count_easy = int(((d == "easy") & (yy == 1)).sum())
            count_medium = int(((d == "medium") & (yy == 1)).sum())
            count_hard = int(((d == "hard") & (yy == 1)).sum())
            print(f"   top 1% {candidate}: easy={count_easy}, "
                  f"medium={count_medium}, hard={count_hard}", flush=True)
        diagnostic.append({
            "seed": seed, "validation_rows": len(va),
            "validation_positives": int(y.iloc[va].sum()),
            "AP": ap,
        })

    mean_ap = {name: float(np.mean([x[name] for x in fold_scores]))
               for name in fold_scores[0]}
    for name, score in sorted(mean_ap.items(), key=lambda item: -item[1]):
        print(f"MEAN_GROUP_AP {name}: {100*score:.2f}", flush=True)
    best = max(mean_ap, key=mean_ap.get)
    # Stability margin: complexity/selection should pay for itself.
    if mean_ap[best] <= mean_ap["v2"] + .002:
        best = "v2"
    print(f"[3/5] Selected '{best}' (grouped AP × 100: {100*mean_ap[best]:.2f})", flush=True)

    print("[4/5] Retrain all base learners on the complete labelled data", flush=True)
    selected_trees = {key: int(np.median([x[key] for x in fold_models]))
                      for key in fold_models[0]}
    fitted, _, _ = train_branches(
        X, y, base_columns, np.arange(len(y)), iterations=selected_trees)

    print("[5/5] Scoring official test set", flush=True)
    test = build_features_v3(a.test)
    assert test.TRANSACTION_HASH.is_unique
    tx_hash = test.TRANSACTION_HASH.copy()
    testX = test.drop(columns=["TRANSACTION_HASH"]).astype("float32")
    assert list(testX.columns) == list(X.columns), "Train/test feature mismatch"
    prediction = candidates(test_predictions(fitted, testX, base_columns))[best]
    prediction = np.clip(np.asarray(prediction), 1e-12, 1-1e-12)
    result = pd.DataFrame({"TRANSACTION_HASH": tx_hash, "score": prediction})
    assert result.score.notna().all() and result.score.between(0, 1).all()
    assert result.TRANSACTION_HASH.is_unique
    if Path(a.test).name.lower() in ("test", "test-data", "decode-and-detect-test-data"):
        assert len(result) == 40101, f"Expected 40101, got {len(result)}"
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest, index=False)
    report = {
        "model_choice": best,
        "seed_list": list(SEEDS[:a.folds]),
        "grouped_ap_mean": mean_ap,
        "fold_diagnostics": diagnostic,
        "full_model_iterations": selected_trees,
        "features": list(X.columns),
        "score_rows": int(len(result)),
        "method": "AI-assisted, label-free graph features, supervised binary target",
        "baseline_official_v2_ap_x100": 52.5
    }
    dest.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(f"DONE: {dest} | {len(result)} scores | candidate {best} | "
          f"grouped CV {mean_ap[best]*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
