"""Retrieval quality (disease-level recall@k, MRR, nDCG) for dense / bm25 / hybrid (+/- crop filter, parent expansion, reranker).
python -m agri.evaluation.retrieval_eval [--rerank]"""
import argparse, time
import pandas as pd
from agri.config import load_config, P, read_jsonl
from agri.evaluation.metrics import retrieval_metrics
from agri.retrieval.core import Retriever

SETS = {"test": "test.jsonl", "handwritten": "handwritten.jsonl", "terminology": "special/terminology.jsonl", "long1024": "special/long1024.jsonl"}

def evaluate(R, rows, k=5, **kw):
    ranked, t0 = [], time.perf_counter()
    for r in rows:
        res = R.search(r["symptom_text"], k=k, parent_expand=False, **kw)["results"]
        ranked.append([c["disease"] for c in res if c["via"] == "direct"])
    m = retrieval_metrics(ranked, [r["diagnosis"] for r in rows]); m["latency_ms"] = (time.perf_counter() - t0) * 1000 / len(rows); return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rerank", action="store_true"); a = ap.parse_args()
    cfg = load_config(); R = Retriever(cfg=cfg); d = P(cfg["paths"]["dataset_dir"]); out = []
    configs = {"dense": dict(mode="dense"), "bm25": dict(mode="bm25"), "hybrid": dict(mode="hybrid"), "hybrid_no_crop_filter": dict(mode="hybrid", crop=None),
               "hybrid_no_alias_expansion": dict(mode="hybrid", expand=False)}
    if a.rerank: configs["hybrid_rerank"] = dict(mode="hybrid", rerank=True)
    for sname, f in SETS.items():
        rows = read_jsonl(d / f)
        for cname, kw in configs.items():
            out.append({"suite": sname, "config": cname, **evaluate(R, rows, **kw)})
    df = pd.DataFrame(out); p = P(cfg["paths"]["results_dir"]) / "retrieval"; p.mkdir(parents=True, exist_ok=True); df.to_csv(p / "retrieval_eval.csv", index=False)
    print(df.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
