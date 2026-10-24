"""SENTINEL V3: label-free wallet-behaviour, graph and batch-flow features.

Uses ONLY the organiser's public transaction edges. Never uses family, difficulty,
scenario_id, cohort_id, split identity, wallet IDs or label values as predictors.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from train_v2 import features_from_edges


def _best_batch(times, amounts, now, target, forward, window, cap=32):
    """Best value-preserving batch made of nearest chronological payments."""
    if target <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    pivot = int(np.searchsorted(times, now, side="right" if forward else "left"))
    if forward:
        end = min(len(times), int(np.searchsorted(times, now + window, side="right")), pivot + cap)
        sample = amounts[pivot:end]
        delays = times[pivot:end] - now
    else:
        begin = max(0, int(np.searchsorted(times, now - window, side="left")), pivot - cap)
        sample = amounts[begin:pivot][::-1]
        delays = now - times[begin:pivot][::-1]
    if not len(sample):
        return (0.0, 0.0, 0.0, 0.0)
    cumulative = np.cumsum(sample)
    # A relative score that tolerates variation in conversion spreads and fees.
    match = np.exp(-np.abs(np.log1p(cumulative) - np.log1p(target)))
    k = int(np.argmax(match))
    # Exact best amount preservation, batch size, delay and normalized total.
    return float(match[k]), float(k + 1), float(delays[k]), float(cumulative[-1] / (1.0 + target))


def build_features_v3(folder):
    from payments import load_payments
    d = pd.DataFrame(load_payments(folder))
    d = d.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)
    feat = features_from_edges(d)
    assert np.array_equal(feat.TRANSACTION_HASH.to_numpy(), d.TRANSACTION_HASH.to_numpy())
    src = d.FROM_ADDRESS.astype(str).to_numpy()
    dst = d.TO_ADDRESS.astype(str).to_numpy()
    t = d.TIMESTAMP.to_numpy(dtype=np.int64)
    value = d.VALUE.to_numpy(dtype=np.float64)
    n = len(d)

    # Research-inspired account ratios / medians / transaction-frequency signals.
    for side, addresses in (("s", d.FROM_ADDRESS), ("r", d.TO_ADDRESS)):
        for role, groupcol in (("out", "FROM_ADDRESS"), ("in", "TO_ADDRESS")):
            grouped = d.groupby(groupcol)
            for suffix, series in (
                ("median", grouped.VALUE.median()),
                ("max", grouped.VALUE.max()),
                ("min", grouped.VALUE.min()),
                ("active_days", (d.TIMESTAMP // 86400).groupby(d[groupcol]).nunique()),
                ("active_hours", (d.TIMESTAMP // 3600).groupby(d[groupcol]).nunique()),
            ):
                feat[f"{side}_{role}_{suffix}"] = addresses.map(series).fillna(0).to_numpy()
        feat[f"{side}_in_out_count_ratio"] = (
            np.log1p(feat[f"{side}_in_count"]) - np.log1p(feat[f"{side}_out_count"]))
        feat[f"{side}_active_fraction"] = (
            (feat[f"{side}_in_active_hours"] + feat[f"{side}_out_active_hours"]) /
            (2.0 + (feat[f"{side}_activity_span"] / 3600.0)))
        feat[f"{side}_partner_ratio"] = (
            (feat[f"{side}_in_unique"] + feat[f"{side}_out_unique"]) /
            (1.0 + feat[f"{side}_in_count"] + feat[f"{side}_out_count"]))
        feat[f"{side}_median_ratio"] = (
            np.log1p(feat[f"{side}_out_median"]) -
            np.log1p(feat[f"{side}_in_median"]))
        feat[f"{side}_value_skew"] = (
            np.log1p(feat[f"{side}_out_mean"]) -
            np.log1p(feat[f"{side}_out_median"]))
        feat[f"{side}_frequency"] = (
            (feat[f"{side}_in_count"] + feat[f"{side}_out_count"]) /
            (1.0 + feat[f"{side}_activity_span"] / 3600.0))

    feat["s_out_amount_percentile"] = d.groupby("FROM_ADDRESS").VALUE.rank(pct=True).to_numpy()
    feat["r_in_amount_percentile"] = d.groupby("TO_ADDRESS").VALUE.rank(pct=True).to_numpy()
    feat["relative_sender_median"] = np.log1p(value) - np.log1p(feat.s_out_median)
    feat["relative_receiver_median"] = np.log1p(value) - np.log1p(feat.r_in_median)

    # Pair behaviour, beyond raw count (which was prone to training-split shortcuts).
    pair = d.groupby(["FROM_ADDRESS", "TO_ADDRESS"]).agg(
        median=("VALUE", "median"), mean=("VALUE", "mean"),
        std=("VALUE", "std"), first=("TIMESTAMP", "min"),
        last=("TIMESTAMP", "max"), distinct=("VALUE", "nunique"))
    pair_idx = pd.MultiIndex.from_arrays([d.FROM_ADDRESS, d.TO_ADDRESS],
                                         names=["FROM_ADDRESS", "TO_ADDRESS"])
    for name, series in pair.items():
        feat["pair_" + name] = series.reindex(pair_idx).to_numpy()
    feat["pair_relative_median"] = np.log1p(value) - np.log1p(feat.pair_median)
    feat["pair_variation"] = feat.pair_std / (1 + feat.pair_mean)
    feat["pair_span"] = feat.pair_last - feat.pair_first
    feat["pair_repeat_ratio"] = feat.pair_distinct / (1 + feat.pair_count)

    # Direction-agnostic, anonymised full-week topology.
    graph = nx.DiGraph()
    graph.add_edges_from(zip(src, dst))
    undirected = graph.to_undirected()
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    core = nx.core_number(undirected)
    weak_size = {}
    for component in nx.weakly_connected_components(graph):
        for node in component:
            weak_size[node] = len(component)
    strong_size = {}
    for component in nx.strongly_connected_components(graph):
        for node in component:
            strong_size[node] = len(component)
    for prefix, addresses in (("s", d.FROM_ADDRESS), ("r", d.TO_ADDRESS)):
        feat[f"{prefix}_graph_core"] = addresses.map(core).fillna(0).to_numpy()
        feat[f"{prefix}_component_size"] = addresses.map(weak_size).fillna(1).to_numpy()
        feat[f"{prefix}_strong_component_size"] = addresses.map(strong_size).fillna(1).to_numpy()
        feat[f"{prefix}_reciprocity"] = addresses.map({
            node: (sum(graph.has_edge(neighbor, node) for neighbor in graph.successors(node)) /
                   max(1, graph.out_degree(node)))
            for node in graph.nodes()
        }).fillna(0).to_numpy()
    feat["graph_core_gap"] = feat.s_graph_core - feat.r_graph_core
    feat["graph_connected_cycle"] = (
        (feat.s_strong_component_size > 1) & (feat.r_strong_component_size > 1)).astype(int)
    feat["graph_hub_to_leaf"] = feat.s_out_unique / (1 + feat.r_out_unique)
    feat["graph_leaf_to_hub"] = feat.r_in_unique / (1 + feat.s_in_unique)

    # Chronological batching: value of multiple incoming/outgoing payments can
    # closely match the transaction amount even if no individual leg matches.
    ins, outs = {}, {}
    for address, indices in d.groupby("TO_ADDRESS", sort=False).indices.items():
        ix = np.sort(indices)
        ins[str(address)] = (t[ix], value[ix])
    for address, indices in d.groupby("FROM_ADDRESS", sort=False).indices.items():
        ix = np.sort(indices)
        outs[str(address)] = (t[ix], value[ix])
    batches = {}
    for direction in ("upstream", "downstream"):
        for window in (3600, 21600, 86400):
            batches[(direction, window)] = np.zeros((n, 4), dtype=np.float64)
    empty_hist = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
    for i in range(n):
        prev = ins.get(src[i], empty_hist)
        future = outs.get(dst[i], empty_hist)
        for window in (3600, 21600, 86400):
            batches[("upstream", window)][i] = _best_batch(
                prev[0], prev[1], t[i], value[i], False, window)
            batches[("downstream", window)][i] = _best_batch(
                future[0], future[1], t[i], value[i], True, window)
    for (direction, window), values in batches.items():
        for j, suffix in enumerate(("best_match", "best_count", "best_delay", "total_ratio")):
            feat[f"{direction}_{suffix}_{window}"] = values[:, j]
    for window in (3600, 21600, 86400):
        a = batches[("upstream", window)]
        b = batches[("downstream", window)]
        feat[f"bidirectional_batch_match_{window}"] = a[:, 0] * b[:, 0]
        feat[f"bidirectional_multi_batch_{window}"] = (
            a[:, 0] * b[:, 0] * (a[:, 1] > 1) * (b[:, 1] > 1))
        feat[f"aggregate_or_split_score_{window}"] = np.maximum(
            a[:, 0] * (a[:, 1] > 1), b[:, 0] * (b[:, 1] > 1))
        feat[f"aggregate_split_count_{window}"] = a[:, 1] + b[:, 1]
    feat["relay_amount_matching"] = (
        feat.upstream_best_match_21600 * feat.downstream_best_match_21600 *
        feat.temporal_two_hop_score)
    feat["prior_in_vs_next_out6h"] = (
        feat.prior_in_usd_21600 - feat.next_out_usd_21600)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    assert feat.TRANSACTION_HASH.is_unique
    return feat
