#!/usr/bin/env bash
# Knowledge base -> verified dataset -> calibration -> retrieval evaluation.   (needs internet once, to download the embedding model)
set -euo pipefail; cd "$(dirname "$0")/.."
python -m agri.kb.build
python -m agri.datagen.build_dataset
python -m agri.datagen.verify
python -m agri.evaluation.calibrate
python -m agri.evaluation.retrieval_eval
python -m agri.evaluation.load_test --n 200
# optional chunking x embedding ablation (rebuilds the KB several times): python -m agri.evaluation.ablation
