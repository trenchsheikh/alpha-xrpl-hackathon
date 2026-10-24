# SENTINEL V3 — the contest runbook

V1 official score: 49.6; V2 official score: 52.5 (best accepted run so far). The V3 official score is NOT known until submission. The 10th-place standing reported by team is not a model performance guarantee.

## Windows PowerShell run commands

From C:\Users\User\Downloads\alpha-xrpl-hackathon:

    git pull origin main
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe tests\test_v3_smoke.py
    .\.venv\Scripts\python.exe train_v3.py --train "C:\Users\User\Downloads\decode-and-detect-training-data\train" --test "C:\Users\User\Downloads\decode-and-detect-test-data" --output predictions_v3.csv
    .\.venv\Scripts\python.exe scripts\package_v3.py

Then submit these files from this folder:

- `sentinel_v3_code.zip` as the Code ZIP.
- `predictions_v3.csv` as the Predictions CSV.
- Run name: `SENTINEL V3 - Behaviour Graph Ensemble`.

The ZIP is standard, not ZIP64; includes all feature/model code, dependencies, licence, environment versions, run instructions and AI disclosure. No raw data, labels, private files, virtual environments, generated model or predictions are included.

## Method

All training data is supplied by the organisers; no external labels are used. Score = 100 × average precision. Only `is_suspicious` is used as a training target. `scenario_id`, `cohort_id`, `difficulty` and `family` are not predictive features. The first two select validation groups; difficulty is used only for reporting performance across positive difficulty categories.

V3 reuses V2 full-window wallet and temporal-edge statistics and adds research-inspired account longevity, median/percentile values, normalised transaction frequency and counterparty diversity, graph k-core and component sizes, plus 1-hour, 6-hour and 24-hour value-preserving batch matching. The specialist uses weights based on `pair_count<=2`, observable from the ledger, NOT annotation metadata.

Models: legacy V2 CatBoost on original features; updated CatBoost on V3 features; LightGBM on V3 features; lower-complexity LightGBM expert with extra weight on low-pair-frequency edges. Two scenario/cohort-grouped validation splits with seeds 42 and 137 compare predeclared standalone models and calibrated/rank ensembles. Choose mean grouped AP winner, with V2 fallback if no meaningful uplift. Retrain on all labelled data at cross-fold best tree counts; produce full official test predictions.

Do NOT compare grouped AP directly against the official leaderboard AP; labels/test distributions differ. Do compare V3 grouped AP against its V2 baseline on the same validation folds. Save predictions_v2.csv and its official score as fallback. The test set has exactly 40,101 transactions.

## Outputs

- `predictions_v3.csv`: the exact leaderboard CSV (`TRANSACTION_HASH,score`).
- `predictions_v3.report.json`: fold scores, selected model/blend and seeds.
- `sentinel_v3_code.zip`: reproducible source package for the portal.

## AI and third-party disclosure

OpenAI GPT-6 in ChatGPT assisted with feature engineering, modelling, code generation, GitHub source maintenance, and this documentation. Leonardo Heinemann's 2024 XRPL account-fraud study informed the choice to examine account activity ratios and LightGBM; no weights or labels were transferred from it. External libraries: NumPy, pandas, PyArrow, SciPy, scikit-learn, CatBoost, LightGBM and NetworkX. Only the competition's synthetic XRPL data is used for training, validation and inference. MIT licence in LICENSE.

## Notes

Full seven-day post-event temporal context is allowed by the challenge rules. This is for offline investigation and is not a streaming real-time implementation. Hard-case labels are in some instances explicitly ambiguous, so 100% detection is not guaranteed. V3 must be tested locally with organiser data before trusting any predicted improvement.