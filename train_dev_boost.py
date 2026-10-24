"""dev-boost entrypoint: medium-windows V3.1 features -> ensemble_boost -> predictions CSV.
python train_dev_boost.py --train DATA/train --test DATA/test --output out/predictions_dev_boost.csv [--config best]
Feature builder is pluggable: --features MODULE:FUNC (FUNC(dir) -> DataFrame or (DataFrame, ...))."""
import argparse, importlib, json
from pathlib import Path
import numpy as np, pandas as pd
import ensemble_boost as eb

CONFIGS = {
    "cat1": {**eb.DEFAULT, "members": [("cat", 42)]},
    "cat3": {**eb.DEFAULT, "members": [("cat", 42), ("cat", 7), ("cat", 99)]},
    "best": {**eb.DEFAULT, "members": [("cat", 42)], "drop": ["relay_match6h", "relay_amount_matching"], "drop_hard_pos": True},
    "cat3_lgb": {**eb.DEFAULT, "members": [("cat", 42), ("cat", 7), ("cat", 99), ("lgb", 42)], "weights": [1, 1, 1, 1]},
}

def feats(spec, folder, cache):
    c = Path(cache) if cache else None
    if c and c.exists(): return pd.read_parquet(c)
    mod, fn = spec.split(":"); out = getattr(importlib.import_module(mod), fn)(folder)
    f = out[0] if isinstance(out, tuple) else out
    if c: c.parent.mkdir(parents=True, exist_ok=True); f.to_parquet(c)
    return f

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train", required=True); a.add_argument("--test", required=True)
    a.add_argument("--output", default="out/predictions_dev_boost.csv"); a.add_argument("--config", default="best")
    a.add_argument("--features", default="train_v31:build")
    a.add_argument("--train-cache"); a.add_argument("--test-cache")
    a = a.parse_args(); cfg = CONFIGS[a.config]
    tr = feats(a.features, a.train, a.train_cache).merge(pd.read_csv(Path(a.train) / "labels.csv")[["TRANSACTION_HASH", "is_suspicious", "difficulty"]], on="TRANSACTION_HASH", validate="one_to_one")
    if cfg.get("drop_hard_pos"):  # train only on Easy/Medium positives + all negatives
        tr = tr[~((tr.difficulty == "hard") & (tr.is_suspicious == 1))]
    te = feats(a.features, a.test, a.test_cache)
    cols = [c for c in tr.columns if c not in ("TRANSACTION_HASH", "is_suspicious", "difficulty") and c in te.columns and pd.api.types.is_numeric_dtype(tr[c])]
    s = eb.fit_predict(tr[cols], tr.is_suspicious.astype(int), te[cols], cfg)
    sub = pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": np.clip(s, 0, 1)})
    assert sub.TRANSACTION_HASH.is_unique and sub.score.between(0, 1).all() and sub.score.notna().all()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); sub.to_csv(a.output, index=False)
    print("wrote", a.output, len(sub), "rows; config", a.config, "features", len(cols))
