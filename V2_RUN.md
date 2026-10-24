# SENTINEL V2 — competition runbook

V1 remains in train.py; its official test score is 49.6. V2 is experimental until scored by the organiser.

Run these PowerShell commands from C:\Users\User\Downloads\alpha-xrpl-hackathon:

    git pull origin main
    .\.venv\Scripts\python.exe tests\test_v2_smoke.py
    .\.venv\Scripts\python.exe train_v2.py --train "C:\Users\User\Downloads\decode-and-detect-training-data\train" --test "C:\Users\User\Downloads\decode-and-detect-test-data" --output predictions_v2.csv

train_v2.py reports GROUP_AP_clock and GROUP_AP_no_clock (grouped by scenario and cohort), a legacy random-split diagnostic, and recall of each difficulty in the top 1%. It chooses whether to keep clock features using grouped AP, retrains on all labels, and generates predictions_v2.csv, predictions_v2.cbm and predictions_v2.report.json.

To produce a non-ZIP64 submission archive using PowerShell:

    @'
    from zipfile import ZipFile, ZIP_DEFLATED
    files = ['train.py','train_v2.py','payments.py','requirements.txt','README.md','V2_RUN.md','LICENSE','scripts/organize_uploads.py','tests/test_v2_smoke.py']
    with ZipFile('sentinel_v2_code.zip', 'w', ZIP_DEFLATED, allowZip64=False) as z:
        for path in files: z.write(path)
        assert z.testzip() is None
    print('Wrote sentinel_v2_code.zip')
    '@ | .\.venv\Scripts\python.exe -

Upload sentinel_v2_code.zip and predictions_v2.csv to the organiser portal as a new run. Preserve your V1 submission and check the V2 official score.

Exact rules: offline full-week graph features allowed, use organiser's data only. Labels' scenario_id/cohort_id are used solely to choose grouped validation folds; they are never predictive features. Seeds are 42; CatBoost is the model. AI disclosure: OpenAI GPT-6/ChatGPT assisted with V2 architecture, feature code, validation and documentation on 24 October 2026. No external transaction datasets or labels were used. MIT license.

Limitations: V2 has not yet been trained end-to-end inside ChatGPT because its runtime lacks PyArrow. The local virtual environment must perform the actual dataset run.