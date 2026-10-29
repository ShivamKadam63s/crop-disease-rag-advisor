"""Build SFT data with the SAME prompts and retriever used at inference.
   Mix: rag_frac of samples use retrieved context (grounded target / INSUFFICIENT_CONTEXT if the needed chunks were not retrieved),
   the rest use the baseline (no-context) prompt. OOD-train examples teach refusal.
python -m agri.train.sft [--rag_frac 0.7]"""
import argparse, random
from collections import Counter
from agri.config import load_config, P, read_jsonl, write_jsonl
from agri.prompts.builder import answer_json, build_messages, OOS_ADVICE, INSUF_ADVICE
from agri.retrieval.core import Retriever

def build_sft(rag_frac=0.7, seed=7):
    cfg = load_config(); rng = random.Random(seed); R = Retriever(cfg=cfg); labels = R.labels
    d = P(cfg["paths"]["dataset_dir"])
    rows, stat = [], Counter()
    for r in read_jsonl(d / "train.jsonl"):
        q, dis = r["symptom_text"], r["diagnosis"]
        if rng.random() < rag_frac:
            msgs, ctx = build_messages(q, labels, R.search(q)["results"], cfg["prompt"])
            have = {c["section"]: i for i, c in enumerate(ctx, 1) if c["disease"] == dis}
            if "cultural_control" in have and "chemical_control" in have:
                cites = [f"S{have[s]}" for s in ("symptoms", "cultural_control", "chemical_control") if s in have]
                tgt = answer_json(r["crop"], dis, r["action"], cites, False); stat["rag_grounded"] += 1
            else:
                tgt = answer_json(None, "INSUFFICIENT_CONTEXT", INSUF_ADVICE, [], True); stat["rag_insufficient"] += 1
        else:
            msgs, _ = build_messages(q, labels, None, cfg["prompt"]); tgt = answer_json(r["crop"], dis, r["action"], [], False); stat["baseline"] += 1
        rows.append({"messages": msgs, "target": tgt, "src": r["id"]})
    for r in read_jsonl(d / "ood_train.jsonl"):
        q = r["symptom_text"]; rag = rng.random() < rag_frac
        msgs, _ = build_messages(q, labels, R.search(q)["results"] if rag else None, cfg["prompt"])
        tgt = answer_json(None, "INSUFFICIENT_CONTEXT", INSUF_ADVICE, [], True) if r["category"] == "dose_unsupported" else answer_json(None, "OUT_OF_SCOPE", OOS_ADVICE, [], True)
        rows.append({"messages": msgs, "target": tgt, "src": r["id"]}); stat["ood_" + ("rag" if rag else "base")] += 1
    rng.shuffle(rows); nv = max(20, len(rows) // 10)
    out = P(cfg["paths"]["sft_dir"]); write_jsonl(out / "train.jsonl", rows[nv:]); write_jsonl(out / "val.jsonl", rows[:nv])
    frac = stat["rag_insufficient"] / max(1, stat["rag_insufficient"] + stat["rag_grounded"])
    print(dict(stat), f"| train {len(rows) - nv} val {nv} | retrieval-miss rate in RAG SFT items: {frac:.1%}")
    if frac > 0.25: print("WARNING: >25% of RAG SFT items are INSUFFICIENT_CONTEXT => retrieval is weak (check embedder/backend=hash?).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rag_frac", type=float, default=0.7); build_sft(ap.parse_args().rag_frac)
