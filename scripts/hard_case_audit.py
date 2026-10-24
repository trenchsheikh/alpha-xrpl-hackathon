"""SENTINEL diagnostic: why easy motifs dominate and hard relays are missed.

Run locally against organiser-provided training data:
 python scripts/hard_case_audit.py --train path/to/train

All private training annotations (difficulty, family, scenario_id, cohort_id)
are used exclusively for offline diagnostics; no private annotations are
included in predictive feature matrices or submitted outputs.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features_v3 import build_features_v3


def safe_auc(y, values):
    if int(y.sum()) == 0 or int(y.sum()) == len(y):
        return None
    vals = np.asarray(values, dtype=float)
    if not np.isfinite(vals).all() or len(np.unique(vals)) < 2:
        return None
    return roc_auc_score(y, vals)


def audit(train_dir):
    f = build_features_v3(train_dir)
    labels = pd.read_csv(Path(train_dir) / "labels.csv")
    d = f.merge(labels, on="TRANSACTION_HASH", how="left", validate="one_to_one")
    assert len(f) == len(d) and d.is_suspicious.notna().all()
    y = d.is_suspicious.astype(int)
    private = {"TRANSACTION_HASH", "is_suspicious", "difficulty", "family", "scenario_id", "cohort_id"}
    features = [col for col in f.columns if col not in private]
    print("\n=== ALL TRAINING TRANSACTIONS ===")
    print("Rows:", len(d), "positive:", int(y.sum()))
    print("Annotation columns:", list(labels.columns))
    for field in ["difficulty", "family"]:
        if field in d:
            print(f"\n=== {field} by label ===")
            table = (pd.crosstab(d[field].fillna("(unassigned)"), y,
                                 dropna=False)
                     .rename(columns={0: "benign", 1: "suspicious"}))
            print(table.to_string())

    relay_rows = (d.family.astype(str).str.contains("relay", case=False, na=False)
                  if "family" in d else pd.Series(False, index=d.index))
    if relay_rows.sum() < 300:
        print("\nNo large named relay subset found; analysing all hard-labelled rows.")
        relay_rows = d.difficulty.astype(str).str.contains("hard", case=False, na=False)
    # If hard negatives aren't annotated, use ALL benign records as controls,
    # and clearly distinguish this less-discriminating comparison.
    hard_positive = y.eq(1) & d.difficulty.astype(str).str.contains("hard", case=False, na=False)
    if (relay_rows & y.eq(0)).sum() < 100:
        print("CAUTION: hard-case control groups unavailable; comparison is versus all benign.")
        relay_rows = hard_positive | y.eq(0)
    print("\n=== HARD/RELAY SEPARABILITY — WITHIN ANNOTATED NEGATIVE CONTROLS ===")
    subset = d.loc[relay_rows].copy()
    yy = subset.is_suspicious.astype(int)
    print("Analysis subset rows:", len(subset), "suspicious:", int(yy.sum()),
          "benign:", int(len(subset) - yy.sum()))
    results = []
    for name in features:
        col = pd.to_numeric(subset[name], errors="coerce").fillna(0)
        auc = safe_auc(yy, col)
        if auc is None:
            continue
        results.append((name, float(auc), float(max(auc, 1-auc)),
                        float(col[yy.eq(1)].median()),
                        float(col[yy.eq(0)].median())))
    results.sort(key=lambda row: -row[2])
    print("\nMost distinctive single features (univariate AUC; maximum is orientation-invariant):")
    print(f"{'Feature':39s} {'AUC':>7s} {'best':>7s} {'med+':>12s} {'med-':>12s}")
    for name, auc, best, pos, neg in results[:35]:
        print(f"{name[:39]:39s} {auc:7.3f} {best:7.3f} {pos:12.3g} {neg:12.3g}")

    print("\n=== PAIR COUNT AND FUND-FLOW CHECKS BY DIFFICULTY ===")
    fields = [k for k in ("pair_count", "temporal_three_hop_score",
                        "upstream_best_match_21600", "downstream_best_match_21600",
                        "bidirectional_multi_batch_21600",
                        "s_in_out_count_ratio", "r_in_out_count_ratio",
                        "graph_connected_cycle", "s_graph_core") if k in subset]
    positive = d[y.eq(1)]
    if "difficulty" in positive:
        print(positive.groupby("difficulty")[fields].median().round(4).to_string())
    if "family" in positive:
        print("\nPositive family counts and median graph features:")
        print(positive.groupby("family")[fields].median().round(4).to_string())

    print("\n=== COUNTERFACTUAL HARD RELAY EXAMPLES ===")
    # Use only aggregated annotations; no account identifiers or transaction hashes emitted.
    if "cohort_id" in subset:
        cohort = subset[subset.cohort_id.notna()].groupby("cohort_id").is_suspicious.agg(["count", "sum"])
        print("Annotated cohorts:", len(cohort),
              "with both positive and benign:", int(((cohort["sum"] > 0) &
                                                 (cohort["sum"] < cohort["count"])).sum()))
    print("\nIMPORTANT: These values are exploratory feature diagnostics on training labels.")
    print("They are NOT cross-validated model scores and cannot establish leaderboard performance.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    args = parser.parse_args()
    audit(args.train)


if __name__ == "__main__":
    main()
