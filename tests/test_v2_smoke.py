"""Fast synthetic-data test without organiser dataset or PyArrow."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from train_v2 import features_from_edges

edges = pd.DataFrame([
 {"TRANSACTION_HASH":"a","TIMESTAMP":1000,"FROM_ADDRESS":"A","TO_ADDRESS":"B","VALUE":100.0},
 {"TRANSACTION_HASH":"b","TIMESTAMP":1060,"FROM_ADDRESS":"B","TO_ADDRESS":"C","VALUE":99.0},
 {"TRANSACTION_HASH":"c","TIMESTAMP":1120,"FROM_ADDRESS":"C","TO_ADDRESS":"D","VALUE":98.0},
 {"TRANSACTION_HASH":"d","TIMESTAMP":2000,"FROM_ADDRESS":"E","TO_ADDRESS":"F","VALUE":200.0},
 {"TRANSACTION_HASH":"e","TIMESTAMP":2100,"FROM_ADDRESS":"F","TO_ADDRESS":"G","VALUE":199.0}
])
features = features_from_edges(edges)
assert len(features) == len(edges)
assert features.TRANSACTION_HASH.is_unique
assert not features.drop(columns="TRANSACTION_HASH").isna().any().any()
assert np.isfinite(features.drop(columns="TRANSACTION_HASH").to_numpy()).all()
first = features.set_index("TRANSACTION_HASH").loc["a"]
assert first["temporal_two_hop_score"] > 0
assert first["temporal_three_hop_score"] > 0
assert first["next_out_count_3600"] >= 1
assert first["r_in_count"] == 1
print("V2 smoke test passed:", len(features.columns)-1, "features")
