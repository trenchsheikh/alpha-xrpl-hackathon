# SENTINEL V3.1 — Economic Trace / Bubble Maps

V1 official score 49.6; V2 official score 52.5. V3 official score not submitted by user. V3.1 is exploratory and is **not guaranteed** to reach 80. Do not submit until two-fold grouped-validation performance and medium/hard detection have been evaluated.

## Exact Windows PowerShell commands

Run from C:\Users\User\Downloads\alpha-xrpl-hackathon:

    git pull origin main
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe tests\test_v31_smoke.py
    .\.venv\Scripts\python.exe train_v31.py --train "C:\Users\User\Downloads\decode-and-detect-training-data\train" --test "C:\Users\User\Downloads\decode-and-detect-test-data" --output predictions_v31.csv

After training, local outputs are `predictions_v31.csv`, `predictions_v31.report.json`, and `sentinel_v31_bubble_map.html`. Open the bubble map in your browser, no internet required. Bubble colour = model-estimated investigation priority; links and colours are NOT allegations against real people. The dataset uses fictional synthetic accounts.

Alternatively regenerate a map using an existing model:

    .\.venv\Scripts\python.exe bubble_map_v31.py --test "C:\Users\User\Downloads\decode-and-detect-test-data" --scores predictions_v3.csv --output bubble_map_v3.html

Only prepare a source ZIP after V3.1 passes training and validation:

    .\.venv\Scripts\python.exe scripts\package_v31.py

Portal files: `sentinel_v31_code.zip` (source, <8 MiB non-ZIP64) and `predictions_v31.csv` (40,101 hashes, [0,1]). Source ZIP also includes dependency info, reproduction commands, MIT licence, and AI disclosure.

## Approach and controls

The value-preserving temporal transaction graph links payment A->B with subsequent B->C transfers only if chronology, value similarity and wallet connectivity support tracing. Up to four direct successive hops are scored forward and backward. Separate subset-sum signals detect aggregation and distribution of multiple transfers, even with unrelated transfers interleaved. Clock windows up to 24 hours and full seven-day context are permitted by the rules. All structural signals are unsupervised (they do not inspect labels). The new models compare enhanced graph features against existing V3 features on exactly the same scenario-grouped folds.

Graph-based risk diffusion is a **candidate** postprocessor, selected only if its grouped-validation AP improves; it does not automatically classify every adjacent wallet as suspicious. It uses top-ranked predictions as seeds, propagates through temporally/value-consistent weighted economic edges, downweights hub-like accounts, and caps support. Validation propagation seeds only held-out transactions to prevent label-informed in-sample training predictions from contaminating validation; the full test split has no training labels.

Both grouped-validation Average Precision and medium/hard top-1% recall are printed; the top model is automatically selected based on mean overall AP, with baseline fallback if no >=0.3 point improvement. Because group-holdout score and hidden test score are different, the official score can differ.

All labels are used solely through the allowed binary target (`is_suspicious`) for supervised modelling. `scenario_id`, `cohort_id`, `family`, `difficulty` are not model features. Scenario/cohort only group validation; difficulty only reports recall and hard-case AP diagnostics. There are no external datasets or known suspicious wallets. The academic XRPL paper suggested behavioural ratio features, not external pretrained labels.

## Tests and provenance

GitHub Actions verifies Python syntax, synthetic temporal chains, subset matching, bounded risk propagation and existing model-training APIs. End-to-end organiser-data training must happen on user's local Windows virtualenv; CI never receives contest data. Seeds 42 and 137 used for scenario-group holdouts, full model refit 42. MIT licence. AI assistance: GPT-6/ChatGPT for code, features, tests and research interpretation.

## Next analysis

If medium/hard recall does not improve, inspect the hard-case audit `scripts/hard_case_audit.py` and avoid promoting V3.1 based only on easy-case AP. Existing submissions are retained as fallback.