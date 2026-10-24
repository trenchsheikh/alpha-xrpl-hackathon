"""Verify CatBoost and LightGBM V3 training and blending APIs end to end."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from train_v3 import train_branches, test_predictions, candidates

rng = np.random.default_rng(12)
n = 440
features = pd.DataFrame({
    "log_usd": rng.normal(size=n),
    "pair_count": rng.choice([1., 2., 3., 8.], size=n),
    "temporal_first_hop_delay": rng.uniform(0, 3000, size=n),
    "bidirectional_batch_match_21600": rng.random(n),
    "frequency": rng.random(n) * 10
}, dtype="float32")
logit = (1.8 * features["log_usd"].to_numpy()
         + 1.4 * features["bidirectional_batch_match_21600"].to_numpy()
         - .2 * features["pair_count"].to_numpy())
target = pd.Series((logit + rng.normal(0, 1.5, n) > 0.8).astype(int))
assert target[:330].sum() > 20 and target[330:].sum() > 6
models, validation, rounds = train_branches(
    features, target, list(features.columns[:3]), np.arange(330),
    np.arange(330, n), iterations={p:70 for p in
    ("v2", "v3_cat", "v3_lgb", "v3_expert")})
candidate_scores = candidates(validation)
assert all(np.isfinite(v).all() for v in candidate_scores.values())
assert 0 <= average_precision_score(target.iloc[330:], candidate_scores["rank_four"]) <= 1
full, _, _ = train_branches(
    features, target, list(features.columns[:3]), np.arange(n),
    iterations={p:50 for p in models})
output = candidates(test_predictions(full, features, list(features.columns[:3])))
assert all(len(v) == n and np.isfinite(v).all() for v in output.values())
print("SENTINEL V3 four-model training and ensemble test PASSED")
