"""End-to-end API (Exp-7 style).  uvicorn agri.service:app --port 8000   ->  POST /diagnose {text, variant, use_rag}"""
import json, re, time
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agri.config import P, load_config, load_variants
from agri.generation.backends import GenerationError

app = FastAPI(title="Crop disease advisory API")

@lru_cache(maxsize=1)
def pipeline():
    from agri.orchestrator.graph import Pipeline
    return Pipeline()

class Req(BaseModel):
    text: str = Field(min_length=5, max_length=4000)
    variant: str = "llm_api"
    use_rag: bool = True

def _log(rec):
    d = P(load_config()["paths"]["logs_dir"]); d.mkdir(exist_ok=True)
    with open(d / "requests.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

@app.get("/health")
def health(): return {"status": "ok", "kb_version": pipeline().R.kb_version}

@app.post("/diagnose")
def diagnose(req: Req):
    if req.variant not in load_variants(): raise HTTPException(422, "unknown variant")
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", req.text).strip(); t0 = time.perf_counter()
    try:
        if req.use_rag:
            out = pipeline().run(text, req.variant)
        else:
            from agri.generation.client import resolve_backend
            from agri.generation.parse import parse_output
            from agri.prompts.builder import build_messages
            p = pipeline(); msgs, _ = build_messages(text, p.labels, None, p.cfg["prompt"]); g = resolve_backend(req.variant, p.cfg).generate(msgs, p.cfg["generation"]["max_new_tokens"])
            pr = parse_output(g["text"], p.labels)
            out = {"status": "unverified", "diagnosis": pr["label"], "crop": pr["crop"], "advice": pr["advice"], "sources": [], "variant": req.variant,
                   "note": "No retrieval: answer is NOT grounded in documents.", "latency_ms": (time.perf_counter() - t0) * 1000}
    except GenerationError as e:
        _log({"t": time.time(), "variant": req.variant, "status": "upstream_error"}); raise HTTPException(502, str(e))
    except Exception:
        _log({"t": time.time(), "variant": req.variant, "status": "internal_error"}); raise HTTPException(500, "internal error")
    if out.get("type") == "error":
        _log({"t": time.time(), "variant": req.variant, "status": "upstream_error"}); raise HTTPException(502, "The language model service is unavailable. Please try again later.")
    out.pop("raw_output", None)
    _log({"t": time.time(), "variant": req.variant, "rag": req.use_rag, "status": out["status"], "latency_ms": round(out["latency_ms"], 1), "chars": len(text)})
    return out
