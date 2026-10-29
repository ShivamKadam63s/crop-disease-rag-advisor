"""Calibrate decision.min_dense_score on TRAIN in-scope queries vs OOD-TRAIN queries (disjoint from every eval set).
Dose-only OOD items are excluded (they are in-topic; the dose guard handles them).
python -m agri.evaluation.calibrate"""
import json
import numpy as np
from agri.config import load_config, P, read_jsonl
from agri.retrieval.core import Retriever

def main():
    cfg = load_config(); R = Retriever(cfg=cfg); d = P(cfg["paths"]["dataset_dir"])
    pos = [R.search(r["symptom_text"], k=3)["dense_top"] for r in read_jsonl(d / "train.jsonl")[:200]]
    neg = [R.search(r["symptom_text"], k=3)["dense_top"] for r in read_jsonl(d / "ood_train.jsonl") if r["category"] != "dose_unsupported"]
    best = None
    for t in np.linspace(min(pos + neg), max(pos + neg), 200):
        tpr, tnr = np.mean(np.array(pos) >= t), np.mean(np.array(neg) < t)
        score = 0.5 * (tpr + tnr) if tpr >= 0.95 else 0.5 * (tpr + tnr) - 1
        if best is None or score > best[0]: best = (score, float(t), float(tpr), float(tnr))
    out = {"min_dense_score": round(best[1], 3)}
    (P("artifacts") / "calibration.json").write_text(json.dumps(out))
    rep = {"threshold": out["min_dense_score"], "in_scope_accept_rate": best[2], "ood_reject_rate": best[3], "pos_range": [min(pos), max(pos)], "neg_range": [min(neg), max(neg)]}
    (P(cfg["paths"]["results_dir"]) / "retrieval").mkdir(parents=True, exist_ok=True)
    (P(cfg["paths"]["results_dir"]) / "retrieval" / "calibration.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
