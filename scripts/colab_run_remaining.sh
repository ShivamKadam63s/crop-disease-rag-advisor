#!/usr/bin/env bash
# Finish the GPU part of the study on a Colab T4 in ONE session (free Colab disconnects after ~2 h
# and drops the machine, so everything below is ordered by priority and prints each metrics.csv /
# efficiency json to stdout so results survive in the notebook output).
#
# Use in Colab (Runtime > T4 GPU):   %%writefile run_all.sh   <paste this file>   then   !bash run_all.sh 2>&1 | tee run_all.log
# L is the per-suite sample size; with L=15 the whole script took ~100 min budget on a T4 (NF4/8-bit
# generation is slow: ~23 min per run at full size). Raise L only if you have a longer session.
export PYTHONUTF8=1
cd /content
rm -rf agri_advisor
git clone -q https://gitlab.com/shivamkadam63-group/crop-disease-rag-advisor.git agri_advisor
cd agri_advisor
pip -q install -r requirements.txt 2>&1 | tail -2
pip -q uninstall -y torchao 2>&1 | tail -1      # Colab's preinstalled torchao is too old for peft
python - <<'EOF'
# only needed if your checkout predates the transformers-5 fix (warmup_ratio was removed)
p="agri/train/train.py"; s=open(p).read()
if "warmup_ratio=0.05," in s:
    s=s.replace("warmup_ratio=0.05,","warmup_steps=max(1, round(0.05*math.ceil(len(tr)/(a.bs*a.accum))*a.epochs)),").replace("import argparse, json, time","import argparse, json, math, time")
    open(p,"w").write(s)
EOF
python -m agri.kb.build 2>&1 | tail -1
python -m agri.datagen.build_dataset 2>&1 | tail -1
python -m agri.datagen.verify 2>&1 | tail -1
python -m agri.train.sft 2>&1 | tail -1
python -m agri.evaluation.calibrate 2>&1 | grep -E "threshold|accept|reject"
F='^train |trainable|train_seconds|train_loss|val_loss|peak_gpu|Error|Traceback'
L=15
R() { echo "=== RUN $1 $2 $(date +%T)"; python -m agri.evaluation.run --variant $1 --rag $2 --pipeline raw --suites all --limit $L 2>&1 | grep -E "^\[|Error|Traceback"; cat results/runs/"$1"__rag-"$2"__raw/metrics.csv; }
E() { echo "=== EFFPPL $1 $(date +%T)"; python -m agri.evaluation.efficiency --variant $1 --rag on 2>&1 | grep -E "Error|Traceback"; python -m agri.evaluation.perplexity --variant $1 2>&1 | grep -E "^\{|Error|Traceback"; cat results/efficiency/"$1"__rag-on.json; }
# 1) variants that need no training
R nf4 on
R nf4 off
R int8 on
R int8 off
E fp16
E nf4
E int8
# 2) LoRA stage
python -m agri.train.train --mode lora --epochs 3 --bs 1 --accum 16 2>&1 | grep -E "$F"
python -m agri.train.merge 2>&1 | tail -1
R lora_merged_nf4 on
R lora_merged_nf4 off
R lora_fp16 on
R lora_fp16 off
E lora_fp16
E lora_merged_nf4
echo "=== LORA STAGE DONE $(date +%T)"
# 3) QLoRA stage
python -m agri.train.train --mode qlora --epochs 3 --bs 1 --accum 16 2>&1 | grep -E "$F"
R qlora_nf4 on
R qlora_nf4 off
E qlora_nf4
echo "=== QLORA STAGE DONE $(date +%T)"
# 4) fp16 runs, human-eval sheets, aggregate, graph (decision-layer) runs
R fp16 on
R fp16 off
for r in nf4__rag-on__raw qlora_nf4__rag-on__raw; do python -m agri.evaluation.human sample --run $r --suite handwritten --n 30 2>&1 | tail -1; done
python -m agri.evaluation.aggregate 2>&1 | tail -2
cat results/grid_quality_robustness.csv
echo "=== AGG DONE $(date +%T)"
for v in nf4 qlora_nf4; do python -m agri.evaluation.run --variant $v --pipeline graph --suites test,handwritten,ood --limit $L 2>&1 | grep -E "^\[|Error|Traceback"; cat results/runs/"$v"__rag-on__graph/metrics.csv; done
echo "=== ALL DONE $(date +%T)"
