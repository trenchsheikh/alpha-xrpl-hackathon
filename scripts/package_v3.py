"""Prepare non-ZIP64 contest source ZIP; no labels, data, secrets or predictions."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from importlib.metadata import PackageNotFoundError, version
import platform
import sys

FILES = [
    "README.md", "V2_RUN.md", "V3_RUN.md", "LICENSE",
    "payments.py", "train.py", "train_v2.py", "train_v3.py",
    "features_v3.py", "requirements.txt",
    "scripts/organize_uploads.py", "scripts/package_v3.py",
    "tests/test_v2_smoke.py", "tests/test_v3_smoke.py", "tests/test_v3_train.py",
]
OPTIONAL = ["predictions_v3.report.json"]

def main():
    root = Path(__file__).resolve().parent.parent
    output = root / "sentinel_v3_code.zip"
    missing = [x for x in FILES if not (root / x).is_file()]
    if missing:
        raise SystemExit("Cannot package; missing files: " + ", ".join(missing))
    versions = [
        "Python: " + platform.python_version(),
        "Operating System: " + platform.platform(),
        "Command: python train_v3.py --train /train --test /test --output predictions_v3.csv",
        "AI assistance: OpenAI GPT-6/ChatGPT developed code and architectural ideas.",
        "Dataset: organiser-provided synthetic XRPL benchmark only.",
    ]
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "catboost",
                    "lightgbm", "networkx", "scipy"):
        try:
            versions.append(package + "==" + version(package))
        except PackageNotFoundError:
            versions.append(package + ": NOT INSTALLED")
    with ZipFile(output, "w", ZIP_DEFLATED, allowZip64=False) as z:
        for name in FILES + OPTIONAL:
            if (root / name).is_file():
                z.write(root / name, arcname=name)
        z.writestr("ENVIRONMENT.txt", "\n".join(versions) + "\n")
        assert z.testzip() is None
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    assert output.stat().st_size < 8 * 1024 * 1024, "Contest ZIP exceeds 8 MiB"

if __name__ == "__main__":
    main()
