"""Chunking x embedding ablation: rebuilds the KB for every setting and evaluates retrieval.
python -m agri.evaluation.ablation [--embedders m1,m2] [--sizes 60,120,200]"""
import argparse, copy, time
import pandas as pd
from agri.config import load_config, P, read_jsonl
from agri.evaluation.retrieval_eval import evaluate
from agri.kb.build import build
from agri.retrieval.core import Retriever

PREFIX = {"BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""), "intfloat/multilingual-e5-small": ("query: ", "passage: "),
          "sentence-transformers/all-MiniLM-L6-v2": ("", ""), "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", "")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", default="BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2,intfloat/multilingual-e5-small")
    ap.add_argument("--sizes", default="60,120,200"); ap.add_argument("--backend"); a = ap.parse_args()
    base = load_config(); d = P(base["paths"]["dataset_dir"]); rows = read_jsonl(d / "test.jsonl") + read_jsonl(d / "handwritten.jsonl"); res = []
    for model in a.embedders.split(","):
        for strategy in ("structured", "naive"):
            for size in [int(x) for x in a.sizes.split(",")]:
                cfg = copy.deepcopy(base); qp, pp = PREFIX.get(model, ("", "")); cfg["retrieval"]["embedder"].update({"query_prefix": qp, "passage_prefix": pp, "model": model})
                out = P("artifacts/ablation") / f"{model.split('/')[-1]}_{strategy}_{size}"; t0 = time.time()
                man = build(cfg, strategy=strategy, max_words=size, overlap=size // 6, model=model, backend=a.backend, out=out, quiet=True); bt = time.time() - t0
                R = Retriever(kb_dir=out, cfg=cfg); m = evaluate(R, rows, mode="hybrid")
                res.append({"embedder": model, "chunking": strategy, "max_words": size, "n_chunks": man["n_chunks"], "build_s": bt, **m}); print(res[-1])
    df = pd.DataFrame(res).sort_values("mrr", ascending=False); p = P(base["paths"]["results_dir"]) / "retrieval"; p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "ablation.csv", index=False); print(df.round(3).to_string(index=False)); print("\nBest setting -> copy into configs/config.yaml (kb.strategy, kb.chunk_max_words, retrieval.embedder) and rebuild.")

if __name__ == "__main__":
    main()
