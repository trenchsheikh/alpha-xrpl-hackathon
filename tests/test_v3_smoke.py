"""V3 tiny offline test: feature matrix, graph cycles, batching, ensemble ranking."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from features_v3 import features_from_edges_v3
from train_v3 import candidates

edges = [
    ("a", 100, "A", "B", 60),
    ("b", 110, "K", "B", 39),
    ("c", 150, "B", "C", 99),
    ("d", 190, "C", "D", 98),
    ("e", 200, "D", "A", 97),
    ("f", 210, "X", "X", 1),
    ("g", 300, "X", "Y", 2),
    ("h", 305, "Y", "Z", 2),
    ("i", 315, "Z", "U", 2),
    ("j", 400, "Q", "B", 3)
]
frame = pd.DataFrame(edges, columns=[
    "TRANSACTION_HASH", "TIMESTAMP", "FROM_ADDRESS", "TO_ADDRESS", "VALUE"])
matrix = features_from_edges_v3(frame)
assert len(matrix) == len(edges)
assert len(matrix.columns) > 100
assert matrix.TRANSACTION_HASH.is_unique
assert not matrix.isna().any().any()
assert np.isfinite(matrix.drop(columns=["TRANSACTION_HASH"]).to_numpy()).all()
byid = matrix.set_index("TRANSACTION_HASH")
assert byid.loc["c", "upstream_best_count_3600"] == 2
assert byid.loc["c", "upstream_best_match_3600"] > .95
assert byid.loc["c", "temporal_two_hop_score"] > 0
assert byid.loc["c", "s_graph_core"] >= 1
fake = {"v2": np.linspace(.01, .9, 10),
        "v3_cat": np.linspace(.15, .95, 10),
        "v3_lgb": np.linspace(.05, .8, 10),
        "v3_expert": np.linspace(.4, .01, 10)}
out = candidates(fake)
assert "rank_four" in out
assert all(len(v) == 10 and np.isfinite(v).all() for v in out.values())
print("SENTINEL V3 features and blending smoke test passed:",
      matrix.shape[0], "transactions;", matrix.shape[1]-1, "features")
