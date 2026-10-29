#!/usr/bin/env bash
# SFT data (same prompts + retriever as inference) -> LoRA (merge to FP16) -> QLoRA.  Needs a CUDA GPU.
set -euo pipefail; cd "$(dirname "$0")/.."
python -m agri.train.sft --rag_frac 0.7
python -m agri.train.train --mode lora  --epochs 3
python -m agri.train.merge
python -m agri.train.train --mode qlora --epochs 3
