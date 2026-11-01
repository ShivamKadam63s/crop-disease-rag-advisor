# Deploy and test (Streamlit Community Cloud)

Written for whoever owns the GitHub account that hosts the app (Shivam). Nothing here needs a GPU.

## 0. What is where

| Branch (GitLab) | Purpose |
|---|---|
| `mac-device-support` | Full code, all results, GPU scripts. Root `requirements.txt` is the FULL list (torch, bitsandbytes...). Use this for Kaggle/Colab/local GPU work. |
| `deploy` | Same code, but the root `requirements.txt` is the SLIM cloud list (no torch). Deploy this one. |

Do not swap requirements by hand: `deploy` already has it. Both branches contain `artifacts/kb/` (the committed knowledge-base index, version `ad3be9cfa693`) and `artifacts/calibration.json` (retrieval gate `0.70`), which the cloud app needs at start-up.

## 1. Put the `deploy` branch on GitHub

```bash
git clone https://gitlab.com/shivamkadam63-group/crop-disease-rag-advisor.git agri && cd agri
git checkout deploy
git remote add github https://github.com/<owner>/crop-disease-rag-advisor.git   # empty PUBLIC repo, no README
git push github deploy:main
```

Check before pushing that no secret is tracked (expect only `.env.example`):
```bash
git ls-files | grep "\.env"
```
Check the deployment files (expect `fastembed ad3be9cfa693`, `{"min_dense_score": 0.70}`, 8 listed files):
```bash
python3 -c "import json; m=json.load(open('artifacts/kb/manifest.json')); print(m['embedder_backend'], m['kb_version'])"
cat artifacts/calibration.json
git ls-files artifacts/kb artifacts/calibration.json
```

## 2. Deploy on Streamlit Community Cloud

1. share.streamlit.io, sign in with GitHub, allow access to the repo.
2. **Create app** -> **Deploy a public app from GitHub**.
3. Repository `<owner>/crop-disease-rag-advisor`, branch `main`, main file `app/streamlit_app.py`, pick a custom URL.
4. **Advanced settings**: Python **3.11**. Secrets (your own key; never commit it):
```toml
LLM_API_KEY = "your_real_groq_key"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-20b"
```
5. **Deploy**. Expected: logs install packages, the app opens after 2-5 minutes (first load downloads the embedding model).

Without a key the app starts, but every question ends in a fallback.

## 3. Test the live app (private window)

| Test | Expected |
|---|---|
| Tomato rings example | Answered, Early blight, with sources |
| Open a source card, switch on "Show in full source document" | Cited text is highlighted |
| Hindi-style wheat example | A sensible rust diagnosis or a clear fallback, never a confident wrong one |
| Cotton pink worms example | Answered, Pink bollworm |
| Mancozeb dose example | Fallback of type dose |
| Cow fever example | Fallback of type out of scope |
| "Banana leaves have black streaks" | Fallback (insufficient evidence), not another crop's disease |
| Empty or very short input | A friendly error message |
| Trace expander | Guard, retrieve, draft, review steps listed |

Checked offline with the mock backend at gate 0.70: tomato answered, cotton answered, cow out-of-scope, banana insufficient. The wheat, dose and UI rows must be checked on the live app with the real key.

## 4. Local tests (optional)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # on `deploy` this is the slim list
cp .env.example .env                       # fill LLM_API_KEY yourself
streamlit run app/streamlit_app.py
```
API check: `uvicorn agri.service:app` then open `/docs` and call `/diagnose`.

## 5. If something fails (Manage app -> Logs)

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | The root `requirements.txt` is not the slim one: you deployed the wrong branch. |
| `faiss` install error | Set Python 3.11 in Advanced settings. |
| `FileNotFoundError: artifacts/kb/manifest.json` | `artifacts/kb/` was not pushed. |
| Almost everything falls back | Gate too high: edit `artifacts/calibration.json`. 0.65 accepts more questions but also lets the banana-type question through. Do NOT run `agri.evaluation.calibrate`, it overwrites the file. |
| Only error fallbacks | Wrong key or model name in Secrets. |

## 6. Before the demo

- Open the app 10 minutes earlier (free apps sleep) and check the Groq quota the day before.
- Keep a screen recording as backup.
- Add the live link at the top of the README.

## 7. Honest limitations to state in the report

- The retrieval gate 0.70 was tuned on evaluation data (`agri.evaluation.threshold_sweep`). Report it as a "tuned operating point" next to the train-calibrated 0.706 / 0.731.
- `lora_merged_nf4` is unstable across training runs: it over-refused in 2 of 4 independent trainings (Kaggle n=15: refusal 73-93%; Kaggle batch full-size: test acc 0.36, refusal 61%) and was fine in the other 2 (Colab: test 0.75; Kaggle partial: test 0.81). Merging LoRA into a 4-bit model is therefore unreliable here; do not deploy it. qlora_nf4 was the best variant in all 4 runs (test 0.91-0.93 with RAG on).
- No hosted-model (Groq) evaluation runs; the Groq key is used only for the live app.
- Human evaluation: sheets in `results/human_eval/` need 2+ raters.

## 10. Changes after the live-app review (2026-11-01)

Guard and fallbacks (`agri/orchestrator/intent.py`, `fallback.py`): dose questions are recognised before the crop check (no crop named -> dose fallback);
animal/human-health words force out-of-scope even next to a crop word; unsupported crops (apple, mango, banana, ...) get their own fallback with no
"closest topics"; vague input ("my plant has spots") asks for the crop; the "please verify before release" note is gone from the helpline.
Graph (`graph.py`): Hindi/Hinglish questions are translated to English before retrieval (hosted backends only); the answer can be rewritten in
Hindi/Hinglish/Marathi (`lang.py`, rejected if the rewrite adds a number); "also consider" when a second disease scores within 0.03 of the diagnosis;
no-crop questions whose two best diseases are from different crops ask for the crop; "which diseases are favoured by ..." is answered extractively from
the Favorable-conditions chunks; a confidence badge (retrieval score vs the 0.70 gate) is returned with every answer.
Answer styles: Quick (default, prompt unchanged), Detailed guide, Simple words, Prevention plan, Organic / low-chemical.
App: question dropdown (28 test questions), photo upload (`agri/vision.py`: photo -> editable symptom text -> normal pipeline), report-a-problem feedback,
Monitoring tab (health check, request table with latency/retries/metadata, CSV download). Logs live in `logs/` on the server and reset on redeploy.

Re-run after every change (needs `LLM_API_KEY` in `.env` or the environment):

```bash
python -m tests.regression_live --variant llm_api     # 38 cases, exit code 1 on any failure
python -m tests.regression_live --variant mock --offline   # guard-only, no key
```

Note: the guard and retrieval-tie check also run inside `--pipeline graph` evaluations, so graph numbers recorded before this change (results/kaggle_batch) describe the older guard.
Vision model: `LLM_VISION_MODEL` (default tries qwen/qwen3.8-27b, then llama-4-scout on the same Groq key).
