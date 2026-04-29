#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x ".venv/bin/pytest" ]; then
  PYTEST_BIN=".venv/bin/pytest"
else
  PYTEST_BIN="pytest"
fi

echo "Running core tests..."
"$PYTEST_BIN" tests/test_feature_engineering.py tests/test_kmeans.py -q
