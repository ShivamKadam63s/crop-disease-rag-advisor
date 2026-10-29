import math
import numpy as np

def clf_metrics(gold, pred, classes):
    """gold/pred: lists of label strings (pred may be None / refusal => counted wrong). classes: {label: {group:..}}"""
    from sklearn.metrics import f1_score
    pred = [p if p else "__none__" for p in pred]
    labs = sorted(set(gold))
    per = dict(zip(labs, f1_score(gold, pred, labels=labs, average=None, zero_division=0)))
    head = [v for k, v in per.items() if classes[k]["group"] == "head"]; rare = [v for k, v in per.items() if classes[k]["group"] == "rare"]
    return {"accuracy": float(np.mean([g == p for g, p in zip(gold, pred)])), "macro_f1": float(np.mean(list(per.values()))),
            "head_f1": float(np.mean(head)) if head else None, "rare_f1": float(np.mean(rare)) if rare else None}, per

def retrieval_metrics(ranked_diseases, gold, ks=(1, 3, 5)):
    """ranked_diseases: per query, list of disease names of ranked DIRECT chunks. Disease-level relevance."""
    out = {f"recall@{k}": float(np.mean([g in r[:k] for r, g in zip(ranked_diseases, gold)])) for k in ks}
    rr, nd = [], []
    for r, g in zip(ranked_diseases, gold):
        rr.append(next((1 / (i + 1) for i, d in enumerate(r) if d == g), 0.0))
        k = max(ks); rel = [1 if d == g else 0 for d in r[:k]]
        dcg = sum(x / math.log2(i + 2) for i, x in enumerate(rel)); ideal = sum(1 / math.log2(i + 2) for i in range(min(k, max(1, sum(rel)))))
        nd.append(dcg / ideal if ideal and sum(rel) else 0.0)
    out["mrr"] = float(np.mean(rr)); out[f"ndcg@{max(ks)}"] = float(np.mean(nd)); out["n"] = len(gold)
    return out
