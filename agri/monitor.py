"""Request log for the Monitoring tab: one JSON line per request (question, response, metadata, metrics). CSV export is built from it."""
import json, time
from pathlib import Path
from agri.config import ROOT

LOG = ROOT / "logs/requests.jsonl"
COLS = ["time", "question", "status", "type", "diagnosis", "response", "n_sources", "mode", "lang", "variant", "intent", "attempts", "retries",
        "latency_ms", "gen_ms", "dense_top", "gate", "confidence", "also_consider", "review_reasons", "kb_version"]

def record(question, res, variant, mode, lang, path=LOG):
    cf = res.get("confidence") or {}
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": question, "status": res["status"], "type": res.get("type") or res.get("kind"),
           "diagnosis": res.get("diagnosis"), "response": (res.get("advice") or res.get("message") or "")[:1500], "n_sources": len(res.get("sources") or []),
           "mode": mode, "lang": res.get("lang") or lang, "variant": variant, "intent": res.get("intent"), "attempts": res.get("attempts"),
           "retries": max(0, (res.get("attempts") or 1) - 1), "latency_ms": round(res.get("latency_ms") or 0), "gen_ms": round(res["gen_ms"]) if res.get("gen_ms") else None,
           "dense_top": cf.get("dense_top"), "gate": cf.get("gate"), "confidence": cf.get("level"), "also_consider": "; ".join(res.get("also_consider") or []),
           "review_reasons": "; ".join(res.get("reasons") or []), "kb_version": res.get("kb_version")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row

def load(path=LOG):
    import pandas as pd
    p = Path(path)
    rows = []
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            try: rows.append(json.loads(l))
            except Exception: pass
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df: df[c] = None
    return df[COLS + [c for c in df.columns if c not in COLS]] if len(df) else pd.DataFrame(columns=COLS)
