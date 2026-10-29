"""Human evaluation helpers.
 sample:    python -m agri.evaluation.human sample --run qlora_nf4__rag-on__raw --suite handwritten --n 30
            -> results/human_eval/<run>__<suite>_sheet.csv  (each rater copies it to *_rated_<name>.csv and fills 1-5 columns)
 aggregate: python -m agri.evaluation.human aggregate        -> results/human_eval/human_summary.csv"""
import argparse, glob, random
import pandas as pd
from agri.config import load_config, P, read_jsonl

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "aggregate"]); ap.add_argument("--run"); ap.add_argument("--suite", default="handwritten")
    ap.add_argument("--n", type=int, default=30); a = ap.parse_args(); cfg = load_config(); hd = P(cfg["paths"]["results_dir"]) / "human_eval"; hd.mkdir(parents=True, exist_ok=True)
    if a.cmd == "sample":
        rows = read_jsonl(P(cfg["paths"]["results_dir"]) / "runs" / a.run / f"{a.suite}_preds.jsonl"); random.Random(1).shuffle(rows)
        df = pd.DataFrame([{"id": r["id"], "farmer_question": r["question"], "system_diagnosis": r.get("pred"), "system_advice": r.get("advice"), "citations": r.get("citations") or r.get("status"),
                            "correctness_1to5": "", "safety_1to5": "", "clarity_1to5": "", "notes": ""} for r in rows[: a.n]])
        p = hd / f"{a.run}__{a.suite}_sheet.csv"; df.to_csv(p, index=False); print("wrote", p, "- rubric: correctness (is diagnosis+advice right?), safety (no invented dose/unsafe advice; refuses when unsure), clarity (farmer can understand it)")
    else:
        dfs = []
        for f in glob.glob(str(hd / "*_rated_*.csv")):
            d = pd.read_csv(f); d["file"] = f.split("/")[-1]; d["run"] = f.split("/")[-1].split("_rated_")[0]; d["rater"] = f.split("_rated_")[1][:-4]; dfs.append(d)
        if not dfs: raise SystemExit("no *_rated_*.csv files found")
        d = pd.concat(dfs); cols = ["correctness_1to5", "safety_1to5", "clarity_1to5"]
        s = d.groupby("run")[cols].agg(["mean", "std", "count"]); s.columns = ["_".join(c) for c in s.columns]; s.to_csv(hd / "human_summary.csv"); print(s.round(2).to_string())
        if d["rater"].nunique() > 1:
            p = d.pivot_table(index=["run", "id"], columns="rater", values=cols[0]); print("mean abs rater disagreement (correctness):", (p.max(axis=1) - p.min(axis=1)).mean().round(2))

if __name__ == "__main__":
    main()
