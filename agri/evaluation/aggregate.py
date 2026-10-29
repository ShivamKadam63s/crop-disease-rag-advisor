"""Merge every result file into CSV grids, plots and an auto-written report: python -m agri.evaluation.aggregate"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agri.config import load_config, P

def md(df, nd=3):
    if df.empty: return "_no data_"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].round(nd)
    h = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return h + "\n".join("| " + " | ".join("" if pd.isna(x) else str(x) for x in r) + " |" for r in d.values)

def main():
    cfg = load_config(); R = P(cfg["paths"]["results_dir"]); R.mkdir(exist_ok=True)
    fr = [pd.read_csv(f) for f in glob.glob(str(R / "runs/*/metrics.csv"))]
    if not fr: raise SystemExit("no runs found")
    q = pd.concat(fr, ignore_index=True); q["config"] = q["variant"] + " / rag=" + q["rag"].map({True: "on", False: "off"}) + " / " + q["pipeline"]
    q.drop(columns=["per_class_f1"], errors="ignore").to_csv(R / "grid_quality_robustness.csv", index=False)
    eff = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(R / "efficiency/*.json"))])
    if not eff.empty: eff.to_csv(R / "grid_efficiency.csv", index=False)
    ppl = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(R / "perplexity/*.json"))])
    if not ppl.empty: ppl.to_csv(R / "grid_perplexity.csv", index=False)
    raw = q[q["pipeline"] == "raw"]; pl = []
    t = raw[raw["suite"] == "test"].set_index("config"); hw = raw[raw["suite"] == "handwritten"].set_index("config")
    figs = R / "figures"; figs.mkdir(exist_ok=True)
    if len(t):
        x = np.arange(len(t)); w = 0.27; fig, ax = plt.subplots(figsize=(max(8, len(t) * 0.9), 4.5))
        for i, (col, lab) in enumerate((("accuracy", "accuracy (avg)"), ("head_f1", "head-class F1"), ("rare_f1", "rare-class F1"))): ax.bar(x + (i - 1) * w, t[col].fillna(0), w, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=60, ha="right", fontsize=7); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Average vs head vs rare (test suite)"); fig.tight_layout(); fig.savefig(figs / "avg_vs_rare.png", dpi=140); plt.close(fig)
    L = raw[raw["suite"].isin(["test", "long512", "long1024", "long2048"])]
    if len(L):
        fig, ax = plt.subplots(figsize=(7, 4.5)); order = {"test": 0, "long512": 512, "long1024": 1024, "long2048": 2048}
        for c, g in L.groupby("config"):
            g = g.assign(x=g["suite"].map(order)).sort_values("x"); ax.plot(["short", "512", "1024", "2048"][: len(g)], g["accuracy"], marker="o", label=c)
        ax.set_ylabel("accuracy"); ax.set_title("Accuracy vs prompt length"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(figs / "accuracy_vs_length.png", dpi=140); plt.close(fig)
    o = q[q["suite"] == "ood"].set_index("config")
    if len(o):
        fig, ax = plt.subplots(figsize=(max(8, len(o) * 0.9), 4.5)); ax.bar(range(len(o)), o["refusal_rate"].fillna(0)); ax.set_xticks(range(len(o))); ax.set_xticklabels(o.index, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_title("OOD refusal rate (higher is safer)"); fig.tight_layout(); fig.savefig(figs / "ood_refusal.png", dpi=140); plt.close(fig)
    if not eff.empty and len(t):
        e = eff.assign(config=eff["variant"] + " / rag=" + eff["rag"].map({True: "on", False: "off"}) + " / raw").merge(t[["accuracy", "rare_f1"]].reset_index(), on="config")
        if "gpu_peak_gb_single" in e and e["gpu_peak_gb_single"].notna().any():
            fig, ax = plt.subplots(figsize=(6, 4.5)); ax.scatter(e["gpu_peak_gb_single"], e["rare_f1"])
            for _, r in e.iterrows(): ax.annotate(r["variant"], (r["gpu_peak_gb_single"], r["rare_f1"]), fontsize=7)
            ax.set_xlabel("peak GPU GB"); ax.set_ylabel("rare-class F1"); fig.tight_layout(); fig.savefig(figs / "memory_vs_rare_f1.png", dpi=140); plt.close(fig)
    # automatic "looks fine on average, fails on the tail" findings
    finds = []
    ref_candidates = [c for c in t.index if c.startswith(("llm_api", "fp16"))]
    for c in t.index:
        r = t.loc[c]
        if pd.notna(r["rare_f1"]) and pd.notna(r["head_f1"]) and r["head_f1"] - r["rare_f1"] > 0.15: finds.append(f"- **{c}**: head F1 {r['head_f1']:.2f} vs rare F1 {r['rare_f1']:.2f} (gap {r['head_f1'] - r['rare_f1']:.2f}); average accuracy {r['accuracy']:.2f} hides it.")
        l = raw[(raw["config"] == c) & (raw["suite"] == "long2048")]
        if len(l) and r["accuracy"] - l["accuracy"].iloc[0] > 0.2: finds.append(f"- **{c}**: accuracy {r['accuracy']:.2f} on short prompts falls to {l['accuracy'].iloc[0]:.2f} at ~2048 tokens.")
        oo = o.loc[c] if c in o.index else None
        if oo is not None and pd.notna(oo["false_answer_rate"]) and oo["false_answer_rate"] > 0.3: finds.append(f"- **{c}**: answers {oo['false_answer_rate']:.0%} of out-of-distribution questions instead of refusing.")
        if pd.notna(r.get("dose_violation_rate")) and r["dose_violation_rate"] > 0: finds.append(f"- **{c}**: unsupported dose numbers in {r['dose_violation_rate']:.0%} of in-scope answers.")
    pivot = lambda col: q.pivot_table(index="config", columns="suite", values=col).reset_index()
    rep = ["# Evaluation report (auto-generated)\n", "## Quality (raw model output; accuracy / macro-F1 / head vs rare)\n", md(q[(q["pipeline"] == "raw") & q["suite"].isin(["test", "handwritten", "terminology"])][["config", "suite", "n", "accuracy", "macro_f1", "head_f1", "rare_f1", "em_crop", "em_both", "parse_ok_rate", "refusal_rate", "grounded_mean", "citation_correct_rate", "dose_violation_rate", "retrieval_hit_rate"]]),
           "\n## Robustness: accuracy by prompt length\n", md(pivot("accuracy")), "\n## Robustness: OOD behaviour\n", md(q[q["suite"] == "ood"][["config", "n", "refusal_rate", "false_answer_rate", "dose_hallucination_rate", "dose_refusal_rate"]]),
           "\n## Decision layer (graph pipeline)\n", md(q[(q["pipeline"] == "graph") & (q["suite"] != "ood")][["config", "suite", "n", "accuracy", "false_accept_rate", "false_fallback_rate", "retry_rate", "citation_correct_rate"]]) if (q["pipeline"] == "graph").any() else "_not run_",
           "\n## Efficiency\n", md(eff) if not eff.empty else "_not run_", "\n## Perplexity (held-out agri text)\n", md(ppl) if not ppl.empty else "_not run_",
           "\n## Automatic findings: where the average hides failures\n", "\n".join(finds) or "_none triggered_"]
    for f, title in (("retrieval/retrieval_eval.csv", "Retrieval"), ("retrieval/ablation.csv", "Chunking x embedding ablation"), ("system/retrieval_load.csv", "Retrieval load test"), ("human_eval/human_summary.csv", "Human evaluation")):
        if (R / f).exists(): rep += [f"\n## {title}\n", md(pd.read_csv(R / f))]
    (R / "REPORT.md").write_text("\n".join(rep)); print("wrote", R / "REPORT.md", "and grids/figures")

if __name__ == "__main__":
    main()
