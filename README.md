# Grounded Crop Disease Diagnosis and Treatment Advisory Using Retrieval-Augmented Language Models

A farmer describes symptoms in plain text; the system returns a likely diagnosis and a **cited** treatment recommendation, or a
helpful fallback when its documents do not support an answer. Retrieval and generation are independent services. Compression
(FP16 / 8-bit / NF4 / LoRA / QLoRA / LoRA-then-NF4) is *studied* against quality, efficiency and robustness, and the evaluation
decides which variant is actually worth shipping.

```
docs -> [KB pipeline] -> chunks + FAISS index + manifest ─┐
user -> Streamlit/API -> LangGraph: guard -> retrieve ─────┼─> [Retrieval service]   (stateless, scale separately)
                                  -> prompt -> draft ──────┼─> [Generation service]  (fp16|int8|nf4|lora|qlora|Groq|Ollama)
                                  -> review -> accept | retry(1x) | fallback
                                  -> answer + clickable citations + trace
eval CLIs call retrieval / generation / graph independently and write CSV + JSONL to results/
```

**Step-by-step guides with expected outputs for every stage (CPU, GPU/Colab, services, deployment, report): `RUN_WINDOWS.md` (PowerShell) and `RUN_MAC_UNIX.md` (Mac/Linux).**

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
cp .env.example .env                                   # then edit: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (Groq: openai/gpt-oss-20b)
bash scripts/smoke_test.sh                             # optional, offline, no GPU/downloads: proves the plumbing (numbers are meaningless)
bash scripts/run_all.sh                                # KB -> verified dataset -> calibration -> training -> full eval -> results/REPORT.md
bash scripts/04_serve.sh app                           # Streamlit UI
```
Run the stages separately if you prefer: `01_prepare.sh` (CPU ok), `02_train.sh` (CUDA GPU), `03_eval_grid.sh`, `04_serve.sh`.
The training stage is sized for a single 16 GB GPU (T4/L4): ~405 SFT examples, 3 epochs, LoRA r=16. *I estimated 25-35 min for both
adapters plus merge; I could not measure it here, so check `results/training/*.json` after the first run.*

## What is where
| Path | Purpose |
|---|---|
| `data/raw_docs/*.md` | Knowledge-base documents (seed disease cards). Add PDF/DOCX/TXT here too. |
| `data/glossary.json` | Crop/disease/chemical aliases, agri lexicon (query expansion, crop filter, intent guard) |
| `agri/kb/` | ingest, clean, parse into cards, structure-aware chunking, validation, embedding, FAISS index, versioned manifest |
| `agri/retrieval/` | `core.py` hybrid retriever, `api.py` FastAPI service, `client.py` remote client, `store.py` FAISS flat/HNSW |
| `agri/prompts/builder.py` | Baseline + RAG + strict-retry prompts (single source of truth for inference, eval and SFT) |
| `agri/generation/` | Backends (HF fp16/8-bit/NF4 +LoRA, OpenAI-compatible/Groq, Ollama, mock), parser, FastAPI service |
| `agri/orchestrator/` | intent guard, review checks, fallbacks, LangGraph flow |
| `agri/datagen/` | deterministic dataset builder + **verifier** |
| `agri/train/` | SFT builder, LoRA/QLoRA trainer, adapter merge |
| `agri/evaluation/` | quality/robustness runner, retrieval eval + ablation, efficiency, perplexity, calibration, load test, human eval, aggregation |
| `agri/service.py` | `POST /diagnose {text, variant, use_rag}` with validation, safe 502, logging |
| `app/streamlit_app.py` | UI with clickable citations, source viewer, trace, feedback, rate limit |

## Knowledge base: read this
* The shipped documents are **seed cards I drafted from general agronomic knowledge** (6 crops, 18 diseases/pests + a safety card). They are *not* copied from ICAR/TNAU
  and are **not verified against them**. Before you present results, replace/extend `data/raw_docs` with real bulletins (PDF/DOCX/TXT work) and fill each card's
  `source_url`. Citations then point to real, checkable documents.
* The seed cards deliberately contain **no numeric doses**; they say "follow the product label". The review step rejects any dose number that is not verbatim in a cited
  chunk, so a missing dose becomes a safe fallback, never an invented number. If you add documents with doses, they pass through verbatim.
* Unstructured PDFs/DOCX are chunked and retrievable (with page numbers) but do not create new diagnosis labels; labels come from structured cards (`## Disease` + `symptoms:` fields).
* `python -m agri.kb.build` validates (missing fields, empty/duplicate chunks, offsets that would break citation highlighting, dose-like strings to review) and aborts on errors.

## Dataset (verified by `python -m agri.datagen.verify`)
410 train / 150 test (+36 hand-written, 40 terminology, 3x30 long-prompt at ~512/1024/2048 tokens, 40 OOD eval, 40 disjoint OOD-train). 8 head classes (60 samples) and 10 rare
classes (8 samples: 5 train / 3 test). Labels, crops and actions come from the KB cards, never from a generator model. The last two symptom statements of each class are test-only.
The verifier checks schema, label/crop consistency, action-to-KB traceability, duplicates, train/test leakage, exact class counts, long-prompt token targets, that filler text contains no
diagnostic words, terminology coverage, and OOD disjointness; it exits non-zero on any error.
**Known limits (also printed by the verifier):** texts are template-composed from KB symptom statements, so the `test` suite is optimistic; use `handwritten` (independent wording, incl. Hindi
transliterations) as the realistic check; rare classes have only 3 test samples, so per-class F1 is noisy.

## Decision layer
guard (domain lexicon) -> retrieve -> weak-retrieval gate (calibrated dense score) -> draft -> review (valid JSON, label in list, crop/label match, every citation exists *and* names the
diagnosed disease, advice supported by cited text, no unsupported dose, retrieval confidence). Fail -> one strict retry (query rewrite, fewer chunks, optional stronger variant via
`decision.escalate_variant`) -> typed fallback (out-of-scope / insufficient evidence with nearest topics and what to tell us / dose / system error) with the helpline.
`python -m agri.evaluation.calibrate` sets the retrieval threshold on train + OOD-train queries, never on eval sets.

## Model choice rule
`results/REPORT.md` compares baseline-prompt vs RAG vs fine-tuned vs quantized on the same suites. If a base (or NF4) model with the RAG prompt meets your thresholds on **rare-class F1,
long prompts, terminology, OOD refusal and dose safety** (not just average accuracy), ship that, or the hosted API / Ollama, and skip fine-tuned models.

## Deploy (Streamlit Community Cloud)
1. After `01_prepare.sh`, commit `artifacts/kb/`, `data/`, `configs/`, `agri/`, `app/`. **Do not commit `.env`.**
2. New app -> main file `app/streamlit_app.py`. The slim `app/requirements.txt` avoids torch (uses `fastembed` ONNX for query embeddings, same BGE model).
   If the platform picks the root `requirements.txt` instead, move/rename it so only the slim one is found.
3. App settings -> Secrets: `LLM_API_KEY = "..."`, `LLM_BASE_URL = "https://api.groq.com/openai/v1"`, `LLM_MODEL = "openai/gpt-oss-20b"`.
4. Free tiers have no GPU and limited RAM; the cloud demo therefore uses the hosted API. Local quantized variants are for the study (`AGRI_ENABLE_LOCAL=1` to expose them in the UI).
5. To scale retrieval and generation separately: `bash scripts/04_serve.sh retrieval|generation`, then set `retrieval.mode` / `generation.mode` to `remote` in `configs/config.yaml`.

## Tested vs not tested (honest status)
Tested offline in the build sandbox: KB build + validation, hybrid retrieval, prompts, parser, LangGraph flow (answer / retry / OOD / dose / error paths), dataset build + verifier, SFT data build,
evaluation runner (raw + graph), aggregation/report, retrieval eval, calibration, load test, both FastAPI services (incl. 502 without key leakage), and the Streamlit app headlessly.
**Not tested (needs downloads/GPU/keys):** the real embedding model, HF/bitsandbytes backends, LoRA/QLoRA training and merge, efficiency/perplexity scripts, live Groq calls.
Expect to fix small environment issues (library versions, bitsandbytes on your CUDA) on first run. Pin versions once it works.

## Troubleshooting
* `8-bit/NF4 variants need a CUDA GPU`: bitsandbytes is GPU-only; run those variants on the GPU machine.
* Groq errors: check `LLM_MODEL` exists in your Groq console; gpt-oss models are reasoning models, so the backend requests `reasoning_effort=low` and a 1024-token budget (`LLM_REASONING_EFFORT` to override).
* Many `INSUFFICIENT_CONTEXT` targets in the SFT build log means retrieval is weak (wrong embedder backend, or KB built with `--backend hash`). Rebuild the KB with the real embedder.
* Update `cost:` prices in `configs/config.yaml` before reading cost numbers.
