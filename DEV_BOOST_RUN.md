# dev-boost run instructions

Features: V3.1 + `medium-windows` (features_v3.py / flow_v31.py, unchanged). Model: CatBoost (seed 42, 600 iters, depth 7, lr .04, l2 6),
relay features `relay_match6h, relay_amount_matching` dropped, hard/ambiguous positives removed from training (Easy/Medium focus).

    python -m pip install -r requirements.txt   # + lightgbm xgboost scipy for ensemble experiments
    python train_dev_boost.py --train <data>/train --test <data>/test --output out/predictions_dev_boost.csv --config best

Runtime ~3 min feature build per split + ~20 s training on 8 CPUs. Python 3.13; catboost 1.2.x, pandas, numpy, scikit-learn, pyarrow, scipy, lightgbm, xgboost.
Grouped CV (5-fold StratifiedGroupKFold on scenario/cohort groups): `python tools/grouped_cv.py --features train_v31:build`,
ensembles `tools/run_experiments.py`, pruning `tools/prune_cv.py`, relabel `tools/relabel_cv.py`, neighbour boost `tools/time_boost_cv.py`.
Labels' scenario/cohort/difficulty are used only for CV grouping and (difficulty) for excluding hard positives from training, never as features.
AI disclosure: code written with AI assistance (Grok Bot) on 24 Oct 2026; organiser data only. MIT licence (LICENSE). Seed 42.
