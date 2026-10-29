#!/usr/bin/env bash
# Full variant x RAG grid + efficiency + perplexity + system-level (graph) runs. All outputs -> results/
# Set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL in .env to include the API reference variant.
set -uo pipefail; cd "$(dirname "$0")/.."
LOCAL="${LOCAL_VARIANTS:-fp16 int8 nf4 lora_fp16 qlora_nf4 lora_merged_nf4}"
API="${API_VARIANTS:-llm_api}"
GRAPH="${GRAPH_VARIANTS:-llm_api qlora_nf4}"
for v in $LOCAL $API; do for rag in on off; do
  echo "=== $v rag=$rag ==="; python -m agri.evaluation.run --variant "$v" --rag "$rag" --pipeline raw --suites all || echo "FAILED $v $rag"
done; done
for v in $LOCAL $API; do python -m agri.evaluation.efficiency --variant "$v" --rag on || echo "FAILED efficiency $v"; done
for v in $LOCAL; do python -m agri.evaluation.perplexity --variant "$v" || echo "FAILED ppl $v"; done
for v in $GRAPH; do python -m agri.evaluation.run --variant "$v" --pipeline graph --suites test,handwritten,terminology,long2048,ood || echo "FAILED graph $v"; done
python -m agri.evaluation.aggregate
