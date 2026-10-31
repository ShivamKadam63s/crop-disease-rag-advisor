"""Retrieval-confidence gate analysis (CPU only, no model needed, ~1 minute).
Shows the distribution of the best dense similarity per suite and what each threshold would accept (in-scope) / reject (OOD).
   python -m agri.evaluation.threshold_sweep
Output: results/retrieval/threshold_sweep.csv, results/figures/threshold_sweep.png

HOW TO USE HONESTLY: the original threshold (artifacts/calibration.json) was calibrated on train + ood_train only.
A threshold chosen here from test/handwritten/ood numbers is a *tuned operating point*: report it as such, next to the
train-calibrated one, never as a clean held-out result."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agri.config import load_config, P, read_jsonl
from agri.retrieval.core import Retriever

IN_SCOPE = {"train": "train.jsonl", "test": "test.jsonl", "handwritten": "handwritten.jsonl", "terminology": "special/terminology.jsonl", "long2048": "special/long2048.jsonl"}
OOD = {"ood": "special/ood.jsonl", "ood_train": "ood_train.jsonl"}

def main():
    cfg = load_config(); R = Retriever(cfg=cfg); d = P(cfg["paths"]["dataset_dir"]); cur = cfg["decision"]["min_dense_score"]
    sc, cat = {}, {}
    for name, f in {**IN_SCOPE, **OOD}.items():
        rows = read_jsonl(d / f)
        if name == "train": rows = rows[:200]
        sc[name] = np.array([R.search(r["symptom_text"], k=3)["dense_top"] for r in rows]); cat[name] = [r.get("category") for r in rows]
    print("Best-dense-score distribution (p5 / median / p95):")
    for n, v in sc.items(): print(f"  {n:12s} n={len(v):3d}  {np.percentile(v, 5):.3f} / {np.median(v):.3f} / {np.percentile(v, 95):.3f}")
    ths = np.round(np.arange(0.45, 0.86, 0.01), 2); rows = []
    for t in ths:
        r = {"threshold": t}
        for n in IN_SCOPE: r[f"accept_{n}"] = float((sc[n] >= t).mean())
        for n in OOD:
            m = np.array([c != "dose_unsupported" for c in cat[n]])        # dose questions are in-topic; the dose guard handles them
            r[f"reject_{n}"] = float((sc[n][m] < t).mean())
        rows.append(r)
    df = pd.DataFrame(rows); out = P(cfg["paths"]["results_dir"]) / "retrieval"; out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "threshold_sweep.csv", index=False)
    show = df[df["threshold"].isin([0.50, 0.55, 0.60, 0.65, 0.70, cur if cur in ths else 0.73, 0.75, 0.80])].drop_duplicates("threshold")
    print("\nAccept rates (in-scope, want high) and reject rates (OOD, want high):"); print(show.round(2).to_string(index=False))
    ok = df[(df["accept_handwritten"] >= 0.90) & (df["accept_test"] >= 0.90)]
    if len(ok):
        b = ok.iloc[-1]; print(f"\nHighest threshold keeping >=90% of handwritten AND test queries: {b['threshold']}  (OOD reject: eval {b['reject_ood']:.2f}, train {b['reject_ood_train']:.2f})")
        print("-> to try it: write {\"min_dense_score\": %s} into artifacts/calibration.json, then rerun the graph runs." % b["threshold"])
    else: print("\nNo threshold keeps 90% of in-scope queries: rely on the review step instead of the retrieval gate.")
    print(f"Current threshold in use: {cur}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in IN_SCOPE: ax.plot(df["threshold"], df[f"accept_{n}"], label=f"accept {n}")
    for n in OOD: ax.plot(df["threshold"], df[f"reject_{n}"], "--", label=f"reject {n}")
    ax.axvline(cur, color="k", lw=0.8, ls=":"); ax.set_xlabel("dense-score threshold"); ax.set_ylabel("rate"); ax.legend(fontsize=7); ax.set_title("Retrieval gate trade-off")
    fig.tight_layout(); (P(cfg["paths"]["results_dir"]) / "figures").mkdir(exist_ok=True); fig.savefig(P(cfg["paths"]["results_dir"]) / "figures" / "threshold_sweep.png", dpi=140)

if __name__ == "__main__":
    main()
