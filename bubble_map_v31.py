"""Offline interactive wallet bubble map for SENTINEL V3.1 predictions.

Bubble colour = estimated *investigation priority*, not guilt. Only money-flow
plausible links are expanded. High-degree hubs do not turn all contacts red.
"""
import argparse
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

from flow_v31 import continuity_graph, csv_graph_data


def create_bubble_map(test_dir, csv_path, output, top=25, max_tx=250, hops=2):
    raw = csv_graph_data(test_dir)
    raw, next_edges, prev_edges = continuity_graph(raw)
    scores = pd.read_csv(csv_path)
    assert set(["TRANSACTION_HASH", "score"]).issubset(scores.columns)
    raw = raw.merge(scores[["TRANSACTION_HASH", "score"]], on="TRANSACTION_HASH",
                    how="left", validate="one_to_one")
    assert raw.score.notna().all(), "Predictions must match all test hashes"
    seeds = raw.nlargest(top, "score").index.to_list()
    chosen = set(seeds)
    q = deque((index, 0) for index in seeds)
    while q and len(chosen) < max_tx:
        i, depth = q.popleft()
        if depth >= hops:
            continue
        candidates = sorted(next_edges[i] + prev_edges[i],
                            key=lambda x: -x[1])
        for j, link_strength in candidates:
            if j in chosen or link_strength < .10:
                continue
            chosen.add(j)
            q.append((j, depth+1))
            if len(chosen) >= max_tx:
                break
    frame = raw.loc[sorted(chosen)].copy()
    G = nx.Graph()
    for row in frame.itertuples():
        G.add_edge(str(row.FROM_ADDRESS), str(row.TO_ADDRESS))
    if len(G) < 2:
        raise RuntimeError("No graph nodes. Check the data and prediction file.")
    rng_seed = 31
    layout = nx.spring_layout(G, seed=rng_seed,
                              iterations=100, k=1.5 / max(1, np.sqrt(len(G))))
    edges = []
    for row in frame.itertuples(index=False):
        u, v = str(row.FROM_ADDRESS), str(row.TO_ADDRESS)
        start, end = layout[u], layout[v]
        edges.append(go.Scatter(
            x=[start[0], end[0]], y=[start[1], end[1]],
            mode="lines", hoverinfo="skip",
            line=dict(width=1.0 + 3.0 * float(row.score),
                      color="#E35D6A" if row.score >= raw.score.quantile(.99)
                      else "#9BB0C8"),
            showlegend=False
        ))
    wallet_volume, wallet_count, wallet_risk, wallet_score = {}, {}, {}, {}
    for row in frame.itertuples():
        for wallet in (str(row.FROM_ADDRESS), str(row.TO_ADDRESS)):
            wallet_count[wallet] = wallet_count.get(wallet, 0) + 1
            wallet_volume[wallet] = wallet_volume.get(wallet, 0.0) + float(row.VALUE)
            wallet_risk[wallet] = max(wallet_risk.get(wallet, 0.0),
                                      float(row.score))
    wallet = list(G.nodes)
    positions = [layout[w] for w in wallet]
    marker_sizes = [13 + 6 * np.sqrt(wallet_count.get(w, 1)) for w in wallet]
    marker_colors = [wallet_risk.get(w, 0) for w in wallet]
    text = [
        "Synthetic wallet: " + w + "<br>Linked transactions in view: " +
        str(wallet_count.get(w, 0)) + "<br>Value shown: $" +
        format(wallet_volume.get(w, 0), ",.2f") +
        "<br>Max incident score: " + format(wallet_risk.get(w, 0), ".4f") +
        "<br><b>A link or high score is not proof of wrongdoing.</b>"
        for w in wallet
    ]
    nodes = go.Scatter(
        x=[p[0] for p in positions], y=[p[1] for p in positions],
        mode="markers", showlegend=False, hovertext=text, hoverinfo="text",
        marker=dict(
            size=marker_sizes, color=marker_colors,
            colorscale=[[0, "#38BFA6"], [.5, "#F1B954"], [1, "#DD486A"]],
            cmin=0, cmax=1, opacity=.96, line=dict(width=1.5, color="#17243B"),
            colorbar=dict(title="Score", thickness=12)
        )
    )
    fig = go.Figure(data=edges + [nodes])
    fig.update_layout(
        title="SENTINEL • XRPL Economic Flow Bubble Map",
        template="plotly_dark", paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220", height=830,
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x"),
        margin=dict(l=12,r=12,t=60,b=30),
        annotations=[dict(
            text="Synthetic wallet network • seed: highest-scored transactions • "
                 "only plausible chronological value-flows expanded • "
                 "colour indicates investigation priority, NOT confirmed criminality",
            showarrow=False, xref="paper", yref="paper",
            x=.5, y=-.03, font=dict(size=11, color="#CBD5E1")
        )]
    )
    path = Path(output)
    plot(fig, filename=str(path), auto_open=False, include_plotlyjs=True)
    print("Saved interactive offline bubble map:", path.resolve())
    print("Wallet bubbles:", len(wallet), "transaction lines:", len(frame))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", default="sentinel_v31_bubble_map.html")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--hops", type=int, default=2)
    args = parser.parse_args()
    create_bubble_map(args.test, args.scores, args.output,
                      top=args.top, hops=args.hops)
