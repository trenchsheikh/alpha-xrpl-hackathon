"""SENTINEL V2 — offline XRPL AML ranking and scenario-aware validation.

Only organiser-supplied data; no private annotation is used as a model feature.
The entire 7-day window may be used for label-free graph feature extraction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEED = 42
WINDOWS = (600, 3600, 21600, 86400)
EMPTY = frozenset()


def features_from_edges(edges):
    """One feature vector per transaction, using only economic payment edges."""
    d = edges[["TRANSACTION_HASH", "TIMESTAMP", "FROM_ADDRESS",
               "TO_ADDRESS", "VALUE"]].copy()
    d["VALUE"] = pd.to_numeric(d["VALUE"], errors="coerce").fillna(0).clip(lower=0)
    d["TIMESTAMP"] = pd.to_numeric(d["TIMESTAMP"], errors="raise").astype("int64")
    assert d.TRANSACTION_HASH.is_unique, "Payments must be one edge per transaction"
    d = d.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)

    src = d.FROM_ADDRESS.astype(str).to_numpy()
    dst = d.TO_ADDRESS.astype(str).to_numpy()
    t = d.TIMESTAMP.to_numpy(dtype=np.int64)
    val = d.VALUE.to_numpy(dtype=np.float64)
    n = len(d)
    f = pd.DataFrame({"TRANSACTION_HASH": d.TRANSACTION_HASH.to_numpy()})
    f["log_usd"] = np.log1p(val)
    f["hour"] = (t // 3600) % 24
    f["time_fraction"] = (t - t.min()) / max(1, int(t.max() - t.min()))
    f["self_payment"] = (src == dst).astype("int8")

    # Wallet statistics: transferable across disjoint train/test wallet IDs.
    sending = d.groupby("FROM_ADDRESS").agg(
        out_count=("VALUE", "size"), out_sum=("VALUE", "sum"),
        out_mean=("VALUE", "mean"), out_std=("VALUE", "std"),
        out_unique=("TO_ADDRESS", "nunique"),
        out_first=("TIMESTAMP", "min"), out_last=("TIMESTAMP", "max"))
    receiving = d.groupby("TO_ADDRESS").agg(
        in_count=("VALUE", "size"), in_sum=("VALUE", "sum"),
        in_mean=("VALUE", "mean"), in_std=("VALUE", "std"),
        in_unique=("FROM_ADDRESS", "nunique"),
        in_first=("TIMESTAMP", "min"), in_last=("TIMESTAMP", "max"))
    for prefix, addresses in (("s", d.FROM_ADDRESS), ("r", d.TO_ADDRESS)):
        for name, values in sending.items():
            f[prefix + "_" + name] = addresses.map(values).fillna(0).to_numpy()
        for name, values in receiving.items():
            f[prefix + "_" + name] = addresses.map(values).fillna(0).to_numpy()
        f[prefix + "_net_ratio"] = np.log1p(f[prefix + "_out_sum"]) - np.log1p(f[prefix + "_in_sum"])
        f[prefix + "_turnover"] = (f[prefix + "_out_sum"] + f[prefix + "_in_sum"]) / (
            1 + np.abs(f[prefix + "_out_sum"] - f[prefix + "_in_sum"]))
        f[prefix + "_activity_span"] = np.maximum(
            f[prefix + "_out_last"], f[prefix + "_in_last"]) - np.where(
                (f[prefix + "_out_count"] > 0) & (f[prefix + "_in_count"] > 0),
                np.minimum(f[prefix + "_out_first"], f[prefix + "_in_first"]),
                np.maximum(f[prefix + "_out_first"], f[prefix + "_in_first"]))
        f[prefix + "_diversity"] = (f[prefix + "_out_unique"] + f[prefix + "_in_unique"]) / (
            1 + f[prefix + "_out_count"] + f[prefix + "_in_count"])
    f["s_relative_amount"] = val / (1 + f["s_out_mean"].to_numpy())
    f["r_relative_amount"] = val / (1 + f["r_in_mean"].to_numpy())
    pairs = d.groupby(["FROM_ADDRESS", "TO_ADDRESS"]).size()
    f["pair_count"] = np.array([pairs.get((a, b), 0) for a, b in zip(src, dst)])
    f["reverse_pair_count"] = np.array([pairs.get((b, a), 0) for a, b in zip(src, dst)])
    f["sender_pair_share"] = f["pair_count"] / (1 + f["s_out_count"])

    # Time-sorted histories: timestamps, amounts, global row indexes, cumulative sum.
    outgoing, incoming = {}, {}
    for account, g in d.groupby("FROM_ADDRESS", sort=False):
        ix = g.index.to_numpy()
        vv = val[ix]
        outgoing[str(account)] = (t[ix], vv, ix, np.r_[0.0, np.cumsum(vv)])
    for account, g in d.groupby("TO_ADDRESS", sort=False):
        ix = g.index.to_numpy()
        vv = val[ix]
        incoming[str(account)] = (t[ix], vv, ix, np.r_[0.0, np.cumsum(vv)])

    for direction, accounts, history, forward in (
        ("prior_in", src, incoming, False),
        ("next_out", dst, outgoing, True),
        ("prior_out", src, outgoing, False),
        ("next_in", dst, incoming, True),
    ):
        counters = {w: np.zeros(n, dtype=np.int32) for w in WINDOWS}
        totals = {w: np.zeros(n, dtype=np.float64) for w in WINDOWS}
        match = np.zeros(n, dtype=np.float64)
        closest_delay = np.full(n, 86401, dtype=np.float64)
        for i, (account, moment) in enumerate(zip(accounts, t)):
            h = history.get(account)
            if h is None:
                continue
            tt, vv, _, cum = h
            pivot = np.searchsorted(tt, moment, side="right" if forward else "left")
            if forward:
                if pivot < len(tt):
                    closest_delay[i] = tt[pivot] - moment
            elif pivot > 0:
                closest_delay[i] = moment - tt[pivot - 1]
            for window in WINDOWS:
                if forward:
                    lo, hi = pivot, np.searchsorted(tt, moment + window, side="right")
                else:
                    lo, hi = np.searchsorted(tt, moment - window, side="left"), pivot
                counters[window][i] = hi - lo
                totals[window][i] = cum[hi] - cum[lo]
                if window == 21600 and hi > lo and val[i] > 0:
                    # Bounded candidate sampling prevents busy exchange hubs dominating runtime.
                    sample = vv[lo:min(lo + 24, hi)] if forward else vv[max(lo, hi - 24):hi]
                    sim = np.minimum(sample, val[i]) / (np.maximum(sample, val[i]) + 1e-9)
                    match[i] = max(match[i], float(sim.max()))
        for window in WINDOWS:
            f[direction + "_count_" + str(window)] = counters[window]
            f[direction + "_usd_" + str(window)] = np.log1p(np.maximum(0, totals[window]))
        f[direction + "_match6h"] = match
        f[direction + "_delay"] = np.minimum(86401, closest_delay)

    f["collect_then_send6h"] = np.minimum(
        1.0, np.exp(f.prior_in_usd_21600.to_numpy()) / (1.0 + val))
    f["send_then_spread6h"] = np.minimum(
        1.0, np.exp(f.next_out_usd_21600.to_numpy()) / (1.0 + val))
    f["relay_match6h"] = (
        f.prior_in_match6h.to_numpy() * f.next_out_match6h.to_numpy())
    f["collect_ratio24h"] = np.log1p(val) - f.prior_in_usd_86400.to_numpy()
    f["distribute_ratio24h"] = f.next_out_usd_86400.to_numpy() - np.log1p(val)

    # Full-window graph topology (no labels, no wallet identity as feature).
    out_neighbors = d.groupby("FROM_ADDRESS").TO_ADDRESS.apply(lambda z: set(map(str, z))).to_dict()
    in_neighbors = d.groupby("TO_ADDRESS").FROM_ADDRESS.apply(lambda z: set(map(str, z))).to_dict()
    f["two_hop_same_endpoints"] = np.array([
        len(out_neighbors.get(a, EMPTY) & in_neighbors.get(b, EMPTY))
        for a, b in zip(src, dst)])
    f["cycle3_potential"] = np.array([
        len(out_neighbors.get(b, EMPTY) & in_neighbors.get(a, EMPTY))
        for a, b in zip(src, dst)])
    f["neighbor_degree_contrast"] = (
        np.log1p(f["s_out_unique"]) - np.log1p(f["r_out_unique"]))
    f["in_out_diversity"] = (
        f["s_in_unique"] + f["r_out_unique"] - f["s_out_unique"] - f["r_in_unique"])

    # Chronological economic paths: edge A->B, then B->C, then C->D.
    hop1 = np.zeros(n)
    hop2 = np.zeros(n)
    hop1_delay = np.full(n, 21601)
    for i, (b, moment, amount) in enumerate(zip(dst, t, val)):
        h = outgoing.get(b)
        if h is None or amount <= 0:
            continue
        tt, vv, indices, _ = h
        lo = np.searchsorted(tt, moment, side="right")
        hi = min(np.searchsorted(tt, moment + 21600, side="right"), lo + 24)
        best_index, best = -1, 0.0
        for j in range(lo, hi):
            ratio = min(float(amount), float(vv[j])) / (max(float(amount), float(vv[j])) + 1e-9)
            weight = ratio / (1.0 + (int(tt[j]) - int(moment)) / 3600.0)
            if weight > best:
                best, best_index = weight, int(indices[j])
                hop1_delay[i] = int(tt[j]) - int(moment)
        hop1[i] = best
        if best_index < 0:
            continue
        c = dst[best_index]
        h2 = outgoing.get(c)
        if h2 is None:
            continue
        t2, v2, _, _ = h2
        cutoff = t[best_index]
        start = np.searchsorted(t2, cutoff, side="right")
        end = min(np.searchsorted(t2, cutoff + 21600, side="right"), start + 24)
        for j in range(start, end):
            ratio = min(float(val[best_index]), float(v2[j])) / (
                max(float(val[best_index]), float(v2[j])) + 1e-9)
            weight = ratio / (1.0 + (int(t2[j]) - int(cutoff)) / 3600.0)
            hop2[i] = max(hop2[i], best * weight)
    f["temporal_two_hop_score"] = hop1
    f["temporal_three_hop_score"] = hop2
    f["temporal_first_hop_delay"] = hop1_delay
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


def build_features(folder):
    from payments import load_payments
    return features_from_edges(pd.DataFrame(load_payments(folder)))


def scenario_groups(labels):
    # Annotation metadata is used only for splitting, never fed to the model.
    # Group hard-case cohorts together, entire labelled motifs together.
    result = []
    for row in labels.itertuples(index=False):
        if pd.notna(row.cohort_id):
            result.append("cohort:" + str(row.cohort_id))
        elif pd.notna(row.scenario_id):
            result.append("scenario:" + str(row.scenario_id))
        else:
            result.append("single:" + str(row.TRANSACTION_HASH))
    return np.asarray(result)


def choose_group_split(y, groups):
    splitter = GroupShuffleSplit(n_splits=12, test_size=.20, random_state=SEED)
    best, best_penalty = None, 1e12
    prevalence = float(y.mean())
    for tr, va in splitter.split(np.arange(len(y)), y, groups):
        if y.iloc[va].sum() < 70 or y.iloc[tr].sum() < 300:
            continue
        penalty = abs(len(va) / len(y) - .2) + 5.0 * abs(float(y.iloc[va].mean()) - prevalence)
        if penalty < best_penalty:
            best, best_penalty = (tr, va), penalty
    if best is None:
        raise RuntimeError("Unable to find group holdout with enough positives")
    return best


def make_model(iterations=1150):
    return CatBoostClassifier(
        iterations=iterations, depth=7, learning_rate=.04, l2_leaf_reg=6,
        random_seed=SEED, loss_function="Logloss", eval_metric="PRAUC",
        verbose=False, thread_count=-1)


def evaluate(features, y, tr, va):
    model = make_model()
    model.fit(features.iloc[tr], y.iloc[tr],
              eval_set=(features.iloc[va], y.iloc[va]),
              early_stopping_rounds=90)
    p = model.predict_proba(features.iloc[va])[:, 1]
    ap = float(average_precision_score(y.iloc[va], p))
    return model, ap, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions_v2.csv")
    parser.add_argument("--skip-random-check", action="store_true")
    args = parser.parse_args()

    print("[1/6] Generating train features", flush=True)
    train_features = build_features(args.train)
    labels = pd.read_csv(Path(args.train) / "labels.csv")
    expected = train_features.TRANSACTION_HASH
    data = train_features.merge(labels, on="TRANSACTION_HASH", how="left",
                                validate="one_to_one")
    assert data.is_suspicious.notna().all(), "Missing training labels"
    assert data.TRANSACTION_HASH.is_unique and len(data) == len(expected)
    y = data.is_suspicious.astype(int).reset_index(drop=True)
    all_features = train_features.drop(columns="TRANSACTION_HASH")
    groups = scenario_groups(data)
    tr, va = choose_group_split(y, groups)
    clocks = ["hour", "time_fraction"]
    reduced = all_features.drop(columns=clocks)
    print("Train", len(y), "positives", int(y.sum()),
          "feature count", len(all_features),
          "grouped validation", len(va), "positives", int(y.iloc[va].sum()), flush=True)

    print("[2/6] Scenario-grouped evaluation (with clock)", flush=True)
    full_model, full_ap, full_pred = evaluate(all_features, y, tr, va)
    print("GROUP_AP_clock:", round(full_ap, 6), "score", round(full_ap * 100, 2), flush=True)

    print("[3/6] Scenario-grouped evaluation (without clock)", flush=True)
    reduced_model, reduced_ap, reduced_pred = evaluate(reduced, y, tr, va)
    print("GROUP_AP_no_clock:", round(reduced_ap, 6),
          "score", round(reduced_ap * 100, 2), flush=True)
    use_clock = full_ap > reduced_ap + .005
    selected = all_features if use_clock else reduced
    selected_model = full_model if use_clock else reduced_model
    selected_pred = full_pred if use_clock else reduced_pred
    selected_ap = full_ap if use_clock else reduced_ap
    print("Selected:", "with clock" if use_clock else "without clock", flush=True)

    if not args.skip_random_check:
        print("[4/6] Random holdout diagnostic (not used for model selection)", flush=True)
        rt, rv = train_test_split(np.arange(len(y)), random_state=SEED,
                                  test_size=.2, stratify=y)
        _, random_ap, _ = evaluate(selected, y, rt, rv)
        print("RANDOM_AP:", round(random_ap, 6),
              "score", round(random_ap * 100, 2), flush=True)
    else:
        random_ap = None

    # The validation labels are for diagnostics only.
    top_n = max(1, int(np.ceil(len(va) * .01)))
    ranked = np.argsort(-selected_pred)[:top_n]
    print("Top 1% grouped holdout positives:",
          int(y.iloc[va].to_numpy()[ranked].sum()), "of", top_n, flush=True)
    details = data.iloc[va].reset_index(drop=True)
    for difficulty in ("easy", "medium", "hard"):
        detected = int(((details.iloc[ranked].difficulty == difficulty) &
                        (details.iloc[ranked].is_suspicious == 1)).sum())
        denominator = int(((details.difficulty == difficulty) &
                           (details.is_suspicious == 1)).sum())
        print("Top-1% recall by difficulty:", difficulty,
              detected, "/", denominator, flush=True)
    print("Top features:", sorted(
        zip(selected.columns, selected_model.feature_importances_),
        key=lambda item: -item[1])[:16], flush=True)

    print("[5/6] Training full model", flush=True)
    iterations = max(100, min(1150, selected_model.best_iteration_ + 1))
    final = make_model(iterations)
    final.fit(selected, y)

    print("[6/6] Scoring test set", flush=True)
    test_features = build_features(args.test)
    test_ids = test_features["TRANSACTION_HASH"]
    assert test_ids.is_unique
    test_matrix = test_features.drop(columns="TRANSACTION_HASH")
    assert set(test_matrix.columns) == set(all_features.columns)
    test_matrix = test_matrix[selected.columns]
    scores = np.clip(final.predict_proba(test_matrix)[:, 1], 0, 1)
    submission = pd.DataFrame({"TRANSACTION_HASH": test_ids,
                               "score": scores})
    assert submission.TRANSACTION_HASH.is_unique
    assert submission.score.notna().all()
    assert submission.score.between(0, 1).all()
    if Path(args.test).name.lower() in ("test", "test-data", "decode-and-detect-test-data"):
        assert len(submission) == 40101, "Official test should have 40,101 hashes"
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(dest, index=False)
    final.save_model(str(dest.with_suffix(".cbm")))
    report = {
        "seed": SEED,
        "train_rows": int(len(y)), "train_positives": int(y.sum()),
        "features": list(selected.columns),
        "scenario_grouped_ap_with_clock": full_ap,
        "scenario_grouped_ap_without_clock": reduced_ap,
        "random_ap_diagnostic": random_ap,
        "selected_grouped_ap": selected_ap,
        "use_clock": use_clock,
        "best_iterations": int(iterations),
        "test_rows": int(len(submission))
    }
    dest.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print("Wrote", str(dest), len(submission), "rows", flush=True)


if __name__ == "__main__":
    main()
