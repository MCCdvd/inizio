#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/backtest.py --output-dir output --no-plot --episodes 5

test -f output/summary.json
test -f output/metrics.csv

echo "Smoke backtest OK"