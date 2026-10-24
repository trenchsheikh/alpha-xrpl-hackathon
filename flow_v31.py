"""SENTINEL economic continuity graph for synthetic XRPL payments.

Build temporal transaction-to-transaction links A->B then B->C only if the
elapsed time and value movements support a possible flow of funds. Edges through
busy hubs are downweighted. Never equate mere adjacency with criminality.
"""
from collections import defaultdict
from bisect import bisect_right
from pathlib import Path
import numpy as np
import pandas as pd


def continuity_graph(frame, hours=24, limit=18):
    """Return (successor edges, predecessor edges) for chronologically ordered tx.
    Each link is (other_index, confidence), with weight based ONLY on raw transaction data.
    """
    d = frame.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)
    t = d.TIMESTAMP.to_numpy(dtype=np.int64)
    value = d.VALUE.to_numpy(dtype=np.float64)
    sender = d.FROM_ADDRESS.astype(str).to_numpy()
    target = d.TO_ADDRESS.astype(str).to_numpy()
    indices = defaultdict(list)
    for i, address in enumerate(sender):
        indices[address].append(i)
    out_times = {key: t[rows] for key, rows in indices.items()}
    outgoing_counts = {key: len(rows) for key, rows in indices.items()}
    # Busy exchange hubs can match arbitrary transfers by chance.
    successors = [[] for _ in range(len(d))]
    predecessors = [[] for _ in range(len(d))]
    horizon = hours * 3600
    for i, (account, moment, v) in enumerate(zip(target, t, value)):
        rows = indices.get(account)
        if not rows or v <= 0:
            continue
        tt = out_times[account]
        low = int(np.searchsorted(tt, moment, side="left"))
        count = 0
        for j in range(low, len(rows)):
            k = rows[j]
            if k <= i:
                continue  # consistent chronology including same-second transfers
            lag = int(t[k] - moment)
            if lag > horizon:
                break
            if count >= limit:
                break
            count += 1
            amount_ratio = min(v, value[k]) / (max(v, value[k]) + 1e-9)
            if amount_ratio < .15:
                continue
            # Retention of value, plausible timing, and degree-aware specificity.
            amount_score = float(amount_ratio ** .70)
            time_score = float(np.exp(-lag / 21600))
            hub_penalty = float((1.0 + np.log1p(outgoing_counts[account])) ** -.6)
            score = amount_score * time_score * hub_penalty
            if score < .055:
                continue
            successors[i].append((k, score))
            predecessors[k].append((i, score))
    return d, successors, predecessors


def _path_features(edges, n, forward):
    # Time-directed DAG; find first/second/third/fourth best economical continuations.
    path = np.zeros((5, n), dtype=np.float64)
    path[0, :] = 1.0
    count = np.zeros(n, dtype=np.int32)
    best_delay = np.full(n, 86401.0)
    order = range(n-1, -1, -1) if forward else range(n)
    for i in order:
        partners = edges[i]
        count[i] = len(partners)
        for k, weight in partners:
            for depth in range(1, 5):
                path[depth, i] = max(path[depth, i],
                                     weight * path[depth-1, k])
    return path, count


def _subset_match(times, values, moment, target, forward, horizon, cap=10):
    """Best one-, two- or three-leg partial trace, order-independent subset.
    Prevents loss of signal when unrelated transactions interrupt a fund flow.
    """
    if target <= 0:
        return np.zeros(4, dtype=float)
    p = int(np.searchsorted(times, moment, side="right" if forward else "left"))
    if forward:
        end = int(np.searchsorted(times, moment+horizon, side="right"))
        candidates = values[p:min(p+cap, end)]
    else:
        low = int(np.searchsorted(times, moment-horizon, side="left"))
        candidates = values[max(low, p-cap):p]
    v = np.asarray(candidates, dtype=float)
    if len(v) == 0:
        return np.zeros(4)
    best1 = float(np.max(np.exp(-np.abs(np.log1p(v)-np.log1p(target)))))
    best2 = 0.0
    best3 = 0.0
    if len(v) >= 2:
        indices = np.triu_indices(len(v), k=1)
        pair_sums = v[indices[0]] + v[indices[1]]
        best2 = float(np.max(np.exp(-np.abs(np.log1p(pair_sums)-np.log1p(target)))))
    if len(v) >= 3:
        a, b, c = np.triu_indices(len(v), k=1)[0], None, None
        triples = []
        # cap=10 -> <=120 triple combinations.
        for x in range(len(v)-2):
            for y in range(x+1, len(v)-1):
                triples.extend((v[x] + v[y] + v[y+1:]).tolist())
        if triples:
            best3 = float(np.max(np.exp(-np.abs(np.log1p(triples)-np.log1p(target)))))
    return np.asarray([best1, best2, best3, max(best2, best3)], dtype=float)


def add_flow_features(frame, features):
    """Pure economic topology and subset matching for a sorted full-window test/train."""
    d, nxt, prv = continuity_graph(frame)
    if not np.array_equal(features.TRANSACTION_HASH.to_numpy(),
                          d.TRANSACTION_HASH.to_numpy()):
        raise ValueError("Feature matrix and transaction graph order mismatch")
    n = len(d)
    t = d.TIMESTAMP.to_numpy(dtype=np.int64)
    value = d.VALUE.to_numpy(dtype=float)
    sender = d.FROM_ADDRESS.astype(str).to_numpy()
    receiver = d.TO_ADDRESS.astype(str).to_numpy()

    forward, fout = _path_features(nxt, n, True)
    backward, bin_ = _path_features(prv, n, False)
    out = features.copy()
    for depth in range(1, 5):
        out[f"fundflow_forward_{depth}"] = forward[depth]
        out[f"fundflow_backward_{depth}"] = backward[depth]
    out["fundflow_matches_next"] = fout
    out["fundflow_matches_previous"] = bin_
    out["fundflow_continuity_2way"] = forward[1] * backward[1]
    out["fundflow_continuity_4hop"] = np.sqrt(forward[2] * backward[2])
    out["fundflow_continuity_6hop"] = np.sqrt(forward[3] * backward[3])
    out["fundflow_possible_middle"] = (
        (forward[1] > .2) & (backward[1] > .2)).astype("int8")
    out["fundflow_longchain"] = np.maximum(forward[4], backward[4])
    # Network "blobs" are not automatically suspicious; describe connected flows,
    # not all transactions in a wallet's component.
    neighbours = [[] for _ in range(n)]
    for i, rows in enumerate(nxt):
        for j, weight in rows:
            if weight >= .10:
                neighbours[i].append(j)
                neighbours[j].append(i)
    visited = np.zeros(n, dtype=bool)
    sizes = np.ones(n, dtype=np.int32)
    sums = np.zeros(n, dtype=float)
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        component = []
        while stack:
            cur = stack.pop()
            component.append(cur)
            for j in neighbours[cur]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
        count = len(component)
        # Do not imply guilt: structural size is used as context.
        if count > 1:
            sizes[component] = count
            sums[component] = np.log1p(value[component].sum())
    out["fundflow_connected_tx_count"] = sizes
    out["fundflow_connected_value_log"] = sums

    incoming, outgoing = defaultdict(list), defaultdict(list)
    for i, (s, r) in enumerate(zip(sender, receiver)):
        incoming[r].append(i)
        outgoing[s].append(i)
    in_hist = {a: (t[ix], value[ix]) for a, ix in incoming.items()}
    out_hist = {a: (t[ix], value[ix]) for a, ix in outgoing.items()}
    blank = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float))
    for direction, addresses, lookup, is_forward in (
        ("prior_input", sender, in_hist, False),
        ("future_output", receiver, out_hist, True),
    ):
        for horizon in (3600, 21600):
            results = np.zeros((n, 4), dtype=float)
            for i in range(n):
                tt, vv = lookup.get(addresses[i], blank)
                results[i] = _subset_match(tt, vv, t[i], value[i], is_forward, horizon)
            for j, suffix in enumerate(("single", "pair", "triple", "multi")):
                out[f"{direction}_{suffix}_{horizon}"] = results[:, j]
    for horizon in (3600, 21600):
        out[f"fundflow_merge_and_split_{horizon}"] = (
            out[f"prior_input_multi_{horizon}"] *
            out[f"future_output_multi_{horizon}"])
        out[f"fundflow_merge_or_split_{horizon}"] = np.maximum(
            out[f"prior_input_multi_{horizon}"],
            out[f"future_output_multi_{horizon}"])
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out, nxt, prv


def risk_diffusion(seed_prob, next_edges, prev_edges, iterations=2,
                   strength=.22):
    """Confidence-capped score support along economic flows.
    Never mark every wallet in a connected component as suspicious.
    Scores in [0,1] are returned without using ground-truth labels.
    """
    scores = np.asarray(seed_prob, dtype=float)
    base = np.clip(scores, 0, 1)
    # Threshold only the strongest preliminary alerts; weak signals do not
    # cascade unchecked. Work in rank space because AP is ranking-based.
    ranks = pd.Series(base).rank(pct=True).to_numpy()
    initial = np.clip((ranks - .94) / .06, 0.0, 1.0)
    support = initial.copy()
    n = len(scores)
    for _ in range(iterations):
        nxt = np.zeros(n, dtype=float)
        for i in range(n):
            if next_edges[i]:
                candidates = sorted((float(w) * support[j] for j, w in next_edges[i]),
                                    reverse=True)[:3]
                nxt[i] += .5 * sum(candidates) / max(1, len(candidates))
            if prev_edges[i]:
                candidates = sorted((float(w) * support[j] for j, w in prev_edges[i]),
                                    reverse=True)[:3]
                nxt[i] += .5 * sum(candidates) / max(1, len(candidates))
        support = np.maximum(initial, np.minimum(.99, nxt))
    return np.clip(support, 0, 1)


def csv_graph_data(dataset_dir):
    from payments import load_payments
    return pd.DataFrame(load_payments(dataset_dir))
