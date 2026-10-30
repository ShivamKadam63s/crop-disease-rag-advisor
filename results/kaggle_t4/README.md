# Kaggle T4 run (2026-10-30/31)

Produced by `scripts/colab_run_remaining.sh` (paths changed to /kaggle/working) on a Kaggle notebook with 2x Tesla T4 (GPU 0 used), per-suite sample size L=15 (n=15 per suite; small, so differences of one item = 6.7 points).
Contents: `runs/` (14 run dirs incl. nf4 and qlora_nf4 graph pipeline), `efficiency/`, `perplexity/`, `training/` (LoRA, QLoRA), `human_eval/` (2 rating sheets, unrated), `figures/`, `grid_*.csv`, `REPORT.md`.
Not covered: llm_api runs (need Groq key), human ratings (need 2+ raters). Training here used bs 1 / accum 16 and differs from the earlier Colab sessions (see results/training/colab_t4_training_summary.md).
Observation: lora_merged_nf4 with RAG on over-refuses (refusal 0.73-0.93 on test/handwritten/terminology); lora_fp16 and qlora_nf4 do not.
