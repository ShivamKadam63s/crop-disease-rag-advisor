# What exists, what is missing (as of 2026-10-30)

Free Colab cut the T4 session at ~2 h (twice) and then refused a GPU ("usage limits"). Re-run `scripts/colab_run_remaining.sh` once GPU quota is back (it is ordered by priority, prints every result to the notebook output, and fits one ~100 min session at L=15).

## Stored and usable
| Checklist item | File |
|---|---|
| 1 KB, 2 dataset, SFT data | `artifacts/kb/`, `data/dataset/`, `data/sft/` (verified, 0 errors; KB version ad3be9cfa693) |
| 3.1-3.4 retrieval eval, calibration, ablation, load test | `results/retrieval/*.csv|json`, `results/system/retrieval_load.csv` (Mac, CPU) |
| 4.1, 4.3 LoRA / QLoRA training reports | `results/training/colab_t4_training_summary.md` (numbers only; adapters died with the Colab VM) |
| Partial grid (full suites, n=150/36/40/30/30/30/40) | `results/colab/grid_stream_part1.txt` (fp16 and lora_fp16, RAG on/off), `grid_stream_part2.txt` (qlora_nf4 RAG on). Summary lines only: accuracy, macro_f1, rare_f1, refusal / false_answer rate. |
| Screenshots | `report_assets/` (01, 02*, 04, 05-10; terminal ones are rendered from real output) |
| Code | branch `mac-device-support` (device detection, MPS/CPU LoRA, transformers-5 fix) |

## Missing (needs GPU quota, or a person)
- `results/runs/*` metrics.csv, grid CSV, figures, REPORT.md: not produced (old mock output moved to `results/_stale_mock/`; do not use).
- Runs not done: nf4, int8, lora_merged_nf4 (both RAG modes), qlora_nf4 RAG off; efficiency (6), perplexity (6), decision-layer graph runs (3); merged model and adapters (re-train: ~9 + ~15 min).
- `llm_api` runs: need a Groq key entered by the owner.
- Human evaluation: sheets can be generated after the runs; ratings need 2+ people.
- Screenshots 03 (architecture diagram), 11-15; real-document KB (1.1); real token lengths (2.7, needs `transformers`, ~1 min).
- Deployment (Streamlit Cloud via GitHub) is the friend's step.


## Update 2026-10-31: Kaggle T4 run completed
The remaining GPU deliverables were run on Kaggle (2x Tesla T4, L=15 per suite) and saved in `results/kaggle_t4/`:
all 6 local variants x RAG on/off (raw), graph runs for nf4 and qlora_nf4, efficiency + perplexity for all 6, LoRA and QLoRA training JSON, unrated human-eval sheets, grids, figures, REPORT.md.
Still open: llm_api runs (Groq key), human ratings (2+ raters), screenshots 03 and 11-15.
Note: n=15 per suite is small; the earlier full-suite Colab lines in `results/colab/` remain the larger-sample evidence for fp16, lora_fp16 and qlora_nf4.


## Update 2026-10-31 (later): full-size Colab run, partial
`results/colab_full/run_full_partial.log` holds the full-suite RAG-on summary lines for all 6 variants (n = 150 test / 36 handwritten / 40 terminology / 40 ood / 30 per long suite). The Colab runtime was cut at ~2 h, so graph runs at gate 0.70, RAG-off full-size runs, efficiency/perplexity and the human sheets for the full-size runs are still missing; per-run CSVs from that VM are gone (only the summary lines were captured).
RAG-on, full suites (test acc / handwritten acc / OOD refusal): qlora_nf4 0.907 / 0.861 / 1.00; lora_fp16 0.867 / 0.778 / 1.00; lora_merged_nf4 0.753 / 0.778 / 0.875; fp16 0.660 / 0.611 / 0.375; int8 0.653 / 0.611 / 0.40; nf4 0.540 / 0.583 / 0.775.


## Update 2026-10-31 (night): second full-size run (Kaggle), also cut short
`results/kaggle_full/run_full_partial.log` (hand-transcribed from the notebook page): all 6 RAG-on full-suite runs, both graph runs at gate 0.70, and part of qlora_nf4 RAG-off. Kaggle's idle timeout killed the session during the first RAG-off run, so there are still no per-run CSVs, efficiency/perplexity files or human sheets for the full-size runs.
Graph pipeline at gate 0.70: qlora_nf4 test 0.727 (refusal 0.22), handwritten 0.806, ood refusal 0.975; nf4 test 0.307 (refusal 0.59), ood refusal 1.0. The decision layer trades accuracy for safety, and with the nf4 model it refuses a lot.
lora_merged_nf4 RAG-on is healthy in this run too (test 0.813): across 3 runs one collapsed (n=15 Kaggle), two did not, so the earlier collapse claim stays retracted.
