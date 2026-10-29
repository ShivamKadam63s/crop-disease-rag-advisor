#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."
bash scripts/01_prepare.sh; bash scripts/02_train.sh; bash scripts/03_eval_grid.sh
echo "Done. See results/REPORT.md, results/grid_*.csv, results/figures/"
