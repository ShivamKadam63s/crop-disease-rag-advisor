#!/usr/bin/env bash
# Offline end-to-end check of ALL non-model code (hash embedder + mock backend; no downloads, no GPU). Numbers are meaningless; it proves the plumbing.
set -euo pipefail; cd "$(dirname "$0")/.."
export AGRI_KB_DIR=artifacts/smoke_kb
python -m agri.kb.build --backend hash --out artifacts/smoke_kb
python -m agri.datagen.build_dataset >/dev/null && python -m agri.datagen.verify | tail -2
python -m agri.train.sft | tail -1
python -m agri.evaluation.calibrate >/dev/null
python -m agri.evaluation.run --variant mock --rag on  --pipeline raw   --suites all --limit 20 | tail -3
python -m agri.evaluation.run --variant mock --rag off --pipeline raw   --suites test,ood --limit 20 | tail -2
python -m agri.evaluation.run --variant mock --pipeline graph --suites test,ood --limit 20 | tail -2
python -m agri.evaluation.retrieval_eval >/dev/null && python -m agri.evaluation.aggregate
rm -rf artifacts/calibration.json artifacts/smoke_kb data/sft results
echo "SMOKE TEST PASSED"
