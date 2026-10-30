# Project Completion Checklist

**Project:** Grounded Crop Disease Diagnosis and Treatment Advisory Using Retrieval-Augmented Language Models

The project is **complete** when every box in Tier 1 is ticked and the validity checks in section 10 pass. Tier 2 makes it strong. Tier 3 is optional polish. Paths are relative to the project root.

---

## 1. Knowledge base (the foundation)

| # | Deliverable | Where | Done when |
|---|---|---|---|
| 1.1 | Source documents with license and source notes | `data/raw_docs/` | Real bulletins added (or the seed cards clearly labelled as seed content in the report) |
| 1.2 | Built index and metadata | `artifacts/kb/` (`chunks.jsonl`, `embeddings.npy`, `faiss.index`, `cards.json`, `labels.json`, `manifest.json`) | All files exist; manifest has a `kb_version` |
| 1.3 | KB validation report | `artifacts/kb/validation_report.json` | `"errors": []`; every warning reviewed |
| 1.4 | Alias glossary | `data/glossary.json` | Crop, disease and chemical aliases present |

## 2. Dataset

| # | Deliverable | Where | Done when |
|---|---|---|---|
| 2.1 | Train and test sets | `data/dataset/train.jsonl`, `test.jsonl` | 410 and 150 rows, 8 head and 10 rare classes |
| 2.2 | Independent hand-written test set | `data/dataset/handwritten.jsonl` | 36 rows |
| 2.3 | Special suites | `data/dataset/special/` (`long512`, `long1024`, `long2048`, `terminology`, `ood`) | 30, 30, 30, 40, 40 rows |
| 2.4 | OOD training set (disjoint from eval) | `data/dataset/ood_train.jsonl` | 40 rows |
| 2.5 | Dataset verification report | `data/dataset/verification_report.json` | `"errors": []` and terminal shows `DATASET VERIFIED: 0 errors` |
| 2.6 | SFT data | `data/sft/train.jsonl`, `val.jsonl` | Retrieval-miss rate under about 15% |
| 2.7 | Real token lengths of long suites | terminal output of runbook step B1 | Three mean-token numbers noted for the report |

## 3. Retrieval

| # | Deliverable | Where | Done when |
|---|---|---|---|
| 3.1 | Retrieval evaluation (recall@k, MRR, nDCG; dense vs BM25 vs hybrid) | `results/retrieval/retrieval_eval.csv` | All four suites present |
| 3.2 | Calibrated confidence threshold | `artifacts/calibration.json`, `results/retrieval/calibration.json` | In-scope accept rate 0.95 or higher |
| 3.3 | Chunking x embedding ablation | `results/retrieval/ablation.csv` | Best setting identified and used (Tier 2) |
| 3.4 | Retrieval load test | `results/system/retrieval_load.csv` | Concurrency 1/4/8/16 rows (local, and remote if you ran Part C) |

## 4. Models and training

| # | Deliverable | Where | Done when |
|---|---|---|---|
| 4.1 | LoRA adapter and training report | `artifacts/models/lora_adapter/`, `results/training/lora.json` | Report has trainable params (about 8.8M, 1.75%), train time, val loss, peak GPU |
| 4.2 | Merged FP16 model | `artifacts/models/lora_merged/` | Folder exists (about 1 GB; not committed to GitHub) |
| 4.3 | QLoRA adapter and training report | `artifacts/models/qlora_adapter/`, `results/training/qlora.json` | Same fields; peak GPU lower than LoRA |
| 4.4 | Training screenshots | `report_assets/` | Both terminals captured |

## 5. Quality, robustness and decision-layer results

One folder per run in `results/runs/<variant>__rag-<on|off>__<raw|graph>/` containing `metrics.csv`, `config.json` and one `<suite>_preds.jsonl` per suite.

| # | Deliverable | Done when |
|---|---|---|
| 5.1 | 12 local raw runs: `fp16`, `int8`, `nf4`, `lora_fp16`, `qlora_nf4`, `lora_merged_nf4`, each with RAG on and off | All 7 suites in each `metrics.csv` |
| 5.2 | 2 API reference raw runs: `llm_api` with RAG on and off | `api_errors` is 0 or empty in every suite |
| 5.3 | 3 full-system (graph) runs: `nf4`, `qlora_nf4`, `llm_api` | Columns `false_accept_rate`, `false_fallback_rate`, `retry_rate` present |
| 5.4 | Merged grid | `results/grid_quality_robustness.csv` has all 17 configurations |

**Metrics that must appear in the grid** (the project brief):

| Brief item | Column or suite |
|---|---|
| Task accuracy | `accuracy` |
| F1 | `macro_f1`, `head_f1`, `rare_f1`; per-class in each `metrics.csv` |
| Exact match | `em_crop`, `em_both` |
| Groundedness and citations | `grounded_mean`, `citation_correct_rate` |
| Dose safety | `dose_violation_rate`, `dose_hallucination_rate` |
| Long prompts | suites `long512`, `long1024`, `long2048` |
| Rare classes | `rare_f1` vs `head_f1` |
| Domain terminology | suite `terminology` |
| Out-of-distribution | suite `ood`: `refusal_rate`, `false_answer_rate` |

## 6. Efficiency and perplexity

| # | Deliverable | Where | Done when |
|---|---|---|---|
| 6.1 | 7 efficiency files (6 local + `llm_api`) | `results/efficiency/*.json`, merged in `results/grid_efficiency.csv` | Fields: `disk_mb`, `gpu_peak_gb_single`, `ttft_ms_p50`, `tpot_ms_mean`, `throughput_tok_s_single`, `energy_wh_per_query`, `cost_usd_per_1k_queries` |
| 6.2 | 6 perplexity files (local variants) | `results/perplexity/*.json`, merged in `results/grid_perplexity.csv` | `perplexity` present |
| 6.3 | Current prices in the cost config | `configs/config.yaml` (`cost:`) | Edited to real values |

Energy, disk size and GPU memory are not meaningful for `llm_api`; report only latency, tokens and cost for it.

## 7. Human evaluation

| # | Deliverable | Where | Done when |
|---|---|---|---|
| 7.1 | Rating sheets for 3 runs (`llm_api`, `nf4`, `qlora_nf4`), 30 questions each | `results/human_eval/*_sheet.csv` | Created |
| 7.2 | Filled sheets from 2 or more raters | `results/human_eval/*_rated_<name>.csv` | Correctness, safety, clarity filled for every row |
| 7.3 | Summary | `results/human_eval/human_summary.csv` | Mean, std, count per run; rater disagreement noted |

## 8. Aggregated report and figures

| # | Deliverable | Where |
|---|---|---|
| 8.1 | Auto-written report with tables and "where the average hides failures" findings | `results/REPORT.md` |
| 8.2 | Figures | `results/figures/avg_vs_rare.png`, `accuracy_vs_length.png`, `ood_refusal.png`, `memory_vs_rare_f1.png` |
| 8.3 | Recommended-variant decision with the supporting rows | In your report conclusion (rule in the runbook, Part D) |
| 8.4 | LangGraph workflow diagram | `docs/workflow_langgraph.mmd` rendered to PNG |

## 9. System, deployment and repository

| # | Deliverable | Done when |
|---|---|---|
| 9.1 | API working | `/diagnose` returns answered, out-of-scope fallback and dose fallback; wrong key gives HTTP 502 with no key leaked |
| 9.2 | Retrieval and generation run as separate services | Part C tested; `retrieval_load.csv` has `remote` rows |
| 9.3 | Streamlit app working locally | Answer, sources, source viewer with highlight, trace, fallbacks, feedback |
| 9.4 | Public deployment | Live URL passes the five example tests in a private browser window |
| 9.5 | Public GitHub repository | Repo link works; `.env` is not in the repository; `artifacts/kb/` is |
| 9.6 | Separate, meaningful commits | `git log --oneline` shows the 10 commits |

## 10. Validity checks (results are only usable if these pass)

| Check | How | Expected |
|---|---|---|
| No mock results left | `Get-ChildItem results\runs` (Windows) or `ls results/runs` | No folder starting with `mock__` |
| Run count | `python -c "import glob; print(len(glob.glob('results/runs/*/metrics.csv')))"` | 17 |
| One knowledge-base version everywhere | `python -c "import glob,json; print({json.load(open(f))['kb_version'] for f in glob.glob('results/runs/*/config.json')})"` | A single value |
| No silent API failures | The B7 check in the runbook | `api_errors` 0 or empty |
| KB built with a real embedder | `python -c "import json; print(json.load(open('artifacts/kb/manifest.json'))['embedder_backend'])"` | `fastembed` or `st`, never `hash` |
| Dataset still verified | `python -m agri.datagen.verify` | `DATASET VERIFIED: 0 errors` |
| Secrets safe | `git ls-files` search for `.env` | Only `.env.example` |

## 11. ESE report (final document)

| Section | Must contain | Source |
|---|---|---|
| 1. Title page | Title, names, course, guide, date, GitHub link, live link | You |
| 2. Index | Table of contents for sections 3 to 7 | Word/Docs auto |
| 3. Abstract | Problem, method, key numbers, headline finding, recommended variant, deployment | Runbook template + `REPORT.md` |
| 4. Introduction | Motivation, hallucination risk, why small/compressed models, objectives, contributions | You |
| 5. Application workflow | Architecture diagram, LangGraph diagram, KB preparation, prompts, decision layer, UI screenshots | Section 8.4, screenshots |
| 6. Results / observations | Retrieval, quality, compression effect, robustness, decision layer, efficiency, perplexity, human evaluation, limitations | `results/` figures and tables |
| 7. Conclusion | Recommended variant with evidence, what retrieval recovers and what it does not, future work, repo and live links | Section 8.3 |

## 12. Completion tiers

**Tier 1: minimum to call the project complete**
- [ ] Sections 1 and 2 (verified KB and dataset)
- [ ] 3.1, 3.2 (retrieval evaluation and calibration)
- [ ] 4.1 to 4.3 (LoRA, merged, QLoRA trained)
- [ ] 5.1 to 5.4 (all 17 runs, grid merged)
- [ ] 6.1, 6.2 (efficiency and perplexity)
- [ ] 8.1, 8.2 (report and figures)
- [ ] 9.3 to 9.5 (app, public deployment, public repo)
- [ ] Section 10 validity checks all pass
- [ ] Section 11 report written with both links

**Tier 2: makes it strong**
- [ ] Human evaluation with two or more raters (section 7)
- [ ] Chunking and embedding ablation (3.3)
- [ ] Separate-services test (9.2) and remote load test
- [ ] Real source documents instead of seed cards (1.1)
- [ ] Real token lengths of the long suites reported (2.7)

**Tier 3: optional polish**
- [ ] Reranker comparison (`python -m agri.evaluation.retrieval_eval --rerank`)
- [ ] Perplexity on the larger Hugging Face agriculture dataset
- [ ] Demo video of the deployed app
- [ ] Repeat runs with different seeds to report variance

## 13. Honest limitations to state in the report

- The seed knowledge base was drafted from general knowledge and is not verified against ICAR or TNAU unless you replaced it.
- Train and test texts are template-built from the knowledge base, so the `test` suite is optimistic; the `handwritten` suite is the realistic number.
- Each rare class has only 3 test items, so per-class F1 is noisy.
- Results come from a single run (no variance estimate), and the human evaluation sample is small.
- Free-tier API limits may have restricted how many API reference samples you ran; state the actual counts (`n` in each `metrics.csv`).
