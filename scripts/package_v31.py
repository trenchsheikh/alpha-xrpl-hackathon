"""Build self-contained V3.1 non-ZIP64 source archive (<8 MiB)."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from importlib.metadata import version, PackageNotFoundError
import platform

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "README.md", "V2_RUN.md", "V3_RUN.md", "V31_RUN.md", "LICENSE",
    "payments.py", "train.py", "train_v2.py", "train_v3.py",
    "train_v31.py", "features_v3.py", "flow_v31.py",
    "bubble_map_v31.py", "requirements.txt",
    "scripts/organize_uploads.py", "scripts/hard_case_audit.py",
    "scripts/package_v3.py", "scripts/package_v31.py",
    "tests/test_v2_smoke.py", "tests/test_v3_smoke.py",
    "tests/test_v3_train.py", "tests/test_v31_smoke.py"
]
OPTIONAL = ["predictions_v31.report.json"]

def main():
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError("Missing source files: " + repr(missing))
    metadata = [
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        "Run: python train_v31.py --train PATH_TO_TRAIN --test PATH_TO_TEST --output predictions_v31.csv",
        "Study: Heinemann (2024), XRP Ledger account-level fraud detection — account ratios and model comparison only.",
        "Disclosure: OpenAI GPT-6/ChatGPT assisted with source code, design, validation and documentation on 2026-10-24.",
        "Only organiser provided synthetic data and labels; no external data or private labels.",
        "License: MIT."
    ]
    for name in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "networkx",
                 "catboost", "lightgbm", "plotly"):
        try: metadata.append(f"{name}=={version(name)}")
        except PackageNotFoundError: metadata.append(f"{name} NOT INSTALLED")
    dest = ROOT / "sentinel_v31_code.zip"
    with ZipFile(dest, "w", ZIP_DEFLATED, allowZip64=False) as z:
        for path in FILES + OPTIONAL:
            if (ROOT / path).is_file(): z.write(ROOT / path, arcname=path)
        z.writestr("ENVIRONMENT.txt", "\n".join(metadata) + "\n")
        assert z.testzip() is None
    assert dest.stat().st_size < 8 * 1024 * 1024
    print("Created", dest, "bytes", dest.stat().st_size)

if __name__ == "__main__":
    main()
