#!/usr/bin/env bash
# Kaggle BATCH run ("Save Version" -> "Save & Run All"): no idle timeout, outputs are kept with the version.
# Notebook = one cell:  !bash -c "$(curl -sL https://gitlab.com/shivamkadam63-group/crop-disease-rag-advisor/-/raw/mac-device-support/scripts/kaggle_batch.sh)" 2>&1 | tee /kaggle/working/run_batch.log
# Work happens in /tmp/work (not saved); only the zip + log land in /kaggle/working. The zip is refreshed after every stage,
# so a timeout still leaves a usable partial result_full.zip. Steps never abort the script.
export PYTHONUTF8=1
OUT=/kaggle/working; [ -d $OUT ] || OUT=/content
W=/tmp/work; mkdir -p $W; cd $W
git clone -q -b mac-device-support https://gitlab.com/shivamkadam63-group/crop-disease-rag-advisor.git agri_advisor
cd agri_advisor
echo "commit: $(git rev-parse --short HEAD)  gate: $(cat artifacts/calibration.json)  gpu: $(nvidia-smi -L | head -1)"
pip -q install -r requirements.txt 2>&1 | tail -2
pip -q uninstall -y torchao 2>&1 | tail -1
python -m agri.datagen.build_dataset 2>&1 | tail -1
python -m agri.datagen.verify 2>&1 | tail -1
python -m agri.train.sft 2>&1 | tail -1
F='^train |trainable|train_seconds|train_loss|val_loss|peak_gpu|Error|Traceback'
pack() { (zip -qr $OUT/result_full.zip results artifacts/calibration.json 2>/dev/null; ls -la $OUT/result_full.zip | awk '{print "zip", $5, "bytes"}'); }
R() { [ -f results/runs/$1__rag-$2__raw/metrics.csv ] && return; echo "=== RUN $1 $2 $(date +%T)"; python -m agri.evaluation.run --variant $1 --rag $2 --pipeline raw --suites all 2>&1 | grep -E "^\[|Error|Traceback"; }
E() { [ -f results/efficiency/$1__rag-on.json ] && [ -f results/perplexity/$1.json ] && return; echo "=== EFFPPL $1 $(date +%T)"; python -m agri.evaluation.efficiency --variant $1 --rag on 2>&1 | grep -E "Error|Traceback"; python -m agri.evaluation.perplexity --variant $1 2>&1 | grep -E "^\{|Error|Traceback"; }
G() { [ -f results/runs/$1__rag-on__graph/metrics.csv ] && return; echo "=== GRAPH $1 $(date +%T)"; python -m agri.evaluation.run --variant $1 --pipeline graph --suites test,handwritten,ood 2>&1 | grep -E "^\[|Error|Traceback"; }
# 1) training
python -m agri.train.train --mode lora --epochs 3 --bs 1 --accum 16 2>&1 | grep -E "$F"
python -m agri.train.merge 2>&1 | tail -1
python -m agri.train.train --mode qlora --epochs 3 --bs 1 --accum 16 2>&1 | grep -E "$F"
echo "=== TRAIN DONE $(date +%T)"; pack
# 2) RAG-on runs (give the per-run CSVs/predictions), best candidates first; human sheets as soon as the runs exist
for v in qlora_nf4 lora_fp16 nf4; do R $v on; done
for r in qlora_nf4__rag-on__raw nf4__rag-on__raw lora_fp16__rag-on__raw; do python -m agri.evaluation.human sample --run $r --suite handwritten --n 30 2>&1 | tail -1; done
pack
for v in fp16 lora_merged_nf4 int8; do R $v on; done
echo "=== RAG-ON DONE $(date +%T)"; pack
# 3) graph runs at the committed gate (0.70)
G qlora_nf4; G nf4
echo "=== GRAPH DONE $(date +%T)"; pack
# 4) RAG-off runs
for v in qlora_nf4 lora_fp16 fp16 nf4 lora_merged_nf4 int8; do R $v off; pack; done
echo "=== RAG-OFF DONE $(date +%T)"
# 5) efficiency + perplexity
for v in qlora_nf4 fp16 nf4 lora_fp16 lora_merged_nf4 int8; do E $v; pack; done
# 6) aggregate (grid, REPORT.md, figures)
python -m agri.evaluation.aggregate 2>&1 | tail -2
pack
echo "=== ALL DONE $(date +%T)"
