#!/usr/bin/env bash
# Usage: tools/eval_branch.sh <source-repo> <branch> <MODULE:FUNC> [extra grouped_cv args]
# Makes a throwaway read-only clone of <branch> (never touches the source repo's worktree), then runs grouped CV.
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd); SRC=$1; BR=$2; SPEC=$3; shift 3
W=/workspace/dev-tester/branches/$(basename "$SRC")-${BR//\//_}
rm -rf "$W"; git clone -q --no-hardlinks --branch "$BR" "$SRC" "$W"
cp -n "$H/payments.py" "$W/" 2>/dev/null || true
cd "$H"; nice -n 10 .venv/bin/python tools/grouped_cv.py --repo "$W" --features "$SPEC" --train data/train --tag "$(basename "$SRC")-${BR//\//_}" "$@"
