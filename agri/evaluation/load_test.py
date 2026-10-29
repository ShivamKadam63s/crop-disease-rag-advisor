"""Retrieval service load test (local or remote per config). python -m agri.evaluation.load_test --n 200"""
import argparse, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from agri.config import load_config, P, read_jsonl
from agri.retrieval.client import get_retriever

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); a = ap.parse_args()
    cfg = load_config(); R = get_retriever(cfg); qs = [r["symptom_text"] for r in read_jsonl(P(cfg["paths"]["dataset_dir"]) / "test.jsonl")][: a.n]; rows = []
    for conc in (1, 4, 8, 16):
        lat = []
        def one(q):
            t = time.perf_counter(); R.search(q); lat.append((time.perf_counter() - t) * 1000)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex: list(ex.map(one, qs))
        wall = time.perf_counter() - t0
        rows.append({"mode": cfg["retrieval"]["mode"], "concurrency": conc, "queries": len(qs), "qps": len(qs) / wall, "p50_ms": np.percentile(lat, 50), "p95_ms": np.percentile(lat, 95)})
    df = pd.DataFrame(rows); p = P(cfg["paths"]["results_dir"]) / "system"; p.mkdir(parents=True, exist_ok=True); df.to_csv(p / "retrieval_load.csv", index=False); print(df.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
