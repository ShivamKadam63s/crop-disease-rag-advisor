#!/usr/bin/env bash
# Full-size (no --limit) GPU re-run on Kaggle / Colab: 12 plain runs, 2 graph runs at the committed gate (0.70),
# 6 efficiency + 6 perplexity files, LoRA/QLoRA training, human-eval sheets, aggregate, results zip.
# Kaggle: Accelerator = GPU T4 x2, Internet = on.   Survives kernel crashes:  !nohup bash run_full.sh > run_full.log 2>&1 &   then  !tail -30 run_full.log
# Steps never abort the script; a failed step prints its error and the next one runs. Order is by priority (RAG-on first).
export PYTHONUTF8=1
W=/kaggle/working; [ -d /kaggle ] || W=/content
cd $W
# resumable: an existing checkout (adapters, finished runs) is kept; delete agri_advisor by hand for a clean start
[ -d agri_advisor/.git ] || git clone -q -b mac-device-support https://gitlab.com/shivamkadam63-group/crop-disease-rag-advisor.git agri_advisor
cd agri_advisor
echo "commit: $(git rev-parse --short HEAD)  gate: $(cat artifacts/calibration.json)"
pip -q install -r requirements.txt 2>&1 | tail -2
pip -q uninstall -y torchao 2>&1 | tail -1       # preinstalled torchao is too old for peft
# NOTE: no kb.build and no calibrate: the committed KB (ad3be9cfa693) and artifacts/calibration.json are used as they are.
python -m agri.datagen.build_dataset 2>&1 | tail -1
python -m agri.datagen.verify 2>&1 | tail -1
python -m agri.train.sft 2>&1 | tail -1
F='^train |trainable|train_seconds|train_loss|val_loss|peak_gpu|Error|Traceback'
R() { [ -f results/runs/$1__rag-$2__raw/metrics.csv ] && { echo "=== SKIP $1 $2 (done)"; return; }; echo "=== RUN $1 $2 $(date +%T)"; python -m agri.evaluation.run --variant $1 --rag $2 --pipeline raw --suites all 2>&1 | grep -E "^\[|Error|Traceback"; }
E() { [ -f results/efficiency/$1__rag-on.json ] && [ -f results/perplexity/$1.json ] && { echo "=== SKIP effppl $1"; return; }; echo "=== EFFPPL $1 $(date +%T)"; python -m agri.evaluation.efficiency --variant $1 --rag on 2>&1 | grep -E "Error|Traceback"; python -m agri.evaluation.perplexity --variant $1 2>&1 | grep -E "^\{|Error|Traceback"; }
G() { [ -f results/runs/$1__rag-on__graph/metrics.csv ] && { echo "=== SKIP graph $1"; return; }; echo "=== GRAPH $1 $(date +%T)"; python -m agri.evaluation.run --variant $1 --pipeline graph --suites test,handwritten,ood 2>&1 | grep -E "^\[|Error|Traceback"; }
# 1) training (adapters are not stored in git)
[ -f results/training/lora.json ] || python -m agri.train.train --mode lora --epochs 3 --bs 1 --accum 16 2>&1 | grep -E "$F"
[ -d artifacts/models/lora_merged ] || python -m agri.train.merge 2>&1 | tail -1
[ -f results/training/qlora.json ] || python -m agri.train.train --mode qlora --epochs 3 --bs 1 --accum 16 2>&1 | grep -E "$F"
echo "=== TRAIN DONE $(date +%T)"
# 2) RAG-on runs, best candidates first
for v in qlora_nf4 lora_fp16 fp16 nf4 lora_merged_nf4 int8; do R $v on; done
echo "=== RAG-ON DONE $(date +%T)"
# 3) graph runs at the final gate
G qlora_nf4; G nf4
# 4) RAG-off runs
for v in qlora_nf4 lora_fp16 fp16 nf4 lora_merged_nf4 int8; do R $v off; done
echo "=== RAG-OFF DONE $(date +%T)"
# 5) efficiency + perplexity
for v in fp16 nf4 int8 lora_fp16 lora_merged_nf4 qlora_nf4; do E $v; done
# 6) human-eval sheets, aggregate, package
for r in qlora_nf4__rag-on__raw nf4__rag-on__raw lora_fp16__rag-on__raw; do python -m agri.evaluation.human sample --run $r --suite handwritten --n 30 2>&1 | tail -1; done
python -m agri.evaluation.aggregate 2>&1 | tail -2
zip -qr $W/results_full.zip results artifacts/calibration.json && ls -la $W/results_full.zip
echo "=== ALL DONE $(date +%T)"
