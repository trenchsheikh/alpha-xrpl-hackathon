# SENTINEL — XRPL Decode & Detect

Competition workspace for the 24 October 2026 synthetic XRP Ledger AML hackathon.

## Run locally

Python 3.11+ required.

```bash
python -m pip install -r requirements.txt
python scripts/organize_uploads.py --source /path/to/downloaded/files --destination data
python train.py --train data/train --test data/test --output predictions.csv
```

The organiser provides three disjoint splits: `train/` (40,131 labelled), `validation/` (unlabelled practice), and **separately** `test/` (40,101 leaderboard transactions). The 3 files originally uploaded without suffixes have different SHA-256 values from both provided train and validation, and are **candidate test files**; verify they belong to the separate test package before competition submission.

If the test files are not yet ready, use `--test data/validation` to check end-to-end predictions; the practice split **cannot** be scored on the leaderboard.

The script prints held-out Average Precision and creates `TRANSACTION_HASH,score`, one row per transaction. No wallet/hash identity inputs, no external labels. Full-window features are permitted by the organiser.

**Data strategy:** This repository is public. Raw Parquet files, labels, and predictions are excluded via `.gitignore`; keep the organiser files in `data/` locally rather than committing >100 MB of data publicly. Use GitHub for versioned code and experiment notes. To execute use the local run command above (GitHub itself does not execute training).

**Current status:** Starter pipeline committed; this environment lacks PyArrow and cannot install it, so actual training score has not been measured here. Run the command on a computer with dependencies installed before trusting performance.

## Competition requirements

- Score: 100 × Average Precision.
- Submission: `TRANSACTION_HASH,score` for exactly 40,101 test hashes, continuous scores in [0,1].
- Final code ZIP must have source, exact run commands, Python/dependency versions, MIT licence, seed, method/disclosure. No data or secrets inside ZIP.
- All experiments done with organiser-supplied data only.

## AI disclosure

OpenAI GPT-6 helped design and write the competition starter using the organiser's documents. Prompts requested an XRPL-native AML scoring pipeline, reverse engineering of temporal and graph patterns, and GitHub workspace setup. Libraries: NumPy, pandas, scikit-learn, PyArrow and CatBoost. No external transaction data, pretrained AML models or hidden labels used.
