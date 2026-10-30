"""Quality / robustness / decision-layer evaluation. All outputs go to results/runs/<run_id>/.
 python -m agri.evaluation.run --variant qlora_nf4 --rag on --pipeline raw --suites all
 --pipeline raw   : retrieve -> prompt -> generate -> parse (measures the MODEL, no accept/retry/fallback)
 --pipeline graph : the full LangGraph system incl. review, retry and fallback (measures the SYSTEM)"""
import argparse, json, time
import numpy as np, pandas as pd
from agri.config import load_config, load_variants, P, read_jsonl, write_jsonl
from agri.evaluation.metrics import clf_metrics
from agri.generation.client import resolve_backend
from agri.generation.parse import parse_output, REFUSALS
from agri.orchestrator.checks import review
from agri.prompts.builder import build_messages
from agri.retrieval.client import get_retriever
from agri.textutils import doses_in, support_score

SUITES = {"test": "test.jsonl", "handwritten": "handwritten.jsonl", "terminology": "special/terminology.jsonl", "long512": "special/long512.jsonl",
          "long1024": "special/long1024.jsonl", "long2048": "special/long2048.jsonl", "ood": "special/ood.jsonl"}

def peak_gpu():
    try:
        from agri import device as dev
        return dev.peak_gb()
    except Exception: return None

def reset_gpu():
    try:
        from agri import device as dev
        dev.reset_peak()
    except Exception: pass

def mean(x):
    x = [v for v in x if v is not None]; return float(np.mean(x)) if x else None

def eval_raw(rows, name, R, backend, cfg, rag, bs):
    labels, pcfg = R.labels, cfg["prompt"]; qs = [r["symptom_text"] for r in rows]
    t0 = time.perf_counter(); rets = [R.search(q) for q in qs] if rag else [None] * len(qs); ret_ms = (time.perf_counter() - t0) * 1000 / len(qs)
    built = [build_messages(q, labels, rt["results"] if rt else None, pcfg) for q, rt in zip(qs, rets)]
    reset_gpu(); t0 = time.perf_counter()
    gens = backend.generate_many([b[0] for b in built], cfg["generation"]["max_new_tokens"], 2 if "long" in name else bs); wall = time.perf_counter() - t0
    out = []
    for r, rt, (msgs, ctx), g in zip(rows, rets, built, gens):
        p = parse_output(g["text"], labels); chk = review(p, ctx, labels, rt["dense_top"] if rt else 1.0, cfg["decision"])
        ctx_doses = set().union(*[doses_in(c["raw_text"]) for c in ctx]) if ctx else set()
        ids = {f"S{i}": c for i, c in enumerate(ctx, 1)}; cited = [ids[c] for c in p["citations"] if c in ids]
        gold = r.get("diagnosis")
        out.append({"id": r["id"], "question": r["symptom_text"][:300], "gold": gold, "gold_crop": r.get("crop"), "category": r.get("category"),
                    "pred": p["label"], "pred_raw": p["diagnosis"], "pred_crop": p["crop"], "parse_ok": p["parse_ok"], "refused": p["label"] in REFUSALS,
                    "correct": p["label"] == gold if gold else None, "crop_ok": (p["crop"] == r.get("crop")) if gold else None,
                    "advice": p["advice"], "citations": p["citations"],
                    "citation_correct": (bool(cited) and len(cited) == len(p["citations"]) and all(c["disease"] == gold for c in cited)) if (gold and rag) else None,
                    "grounded": support_score(p["advice"], [c["raw_text"] for c in (cited or ctx)]) if (rag and p["advice"]) else None,
                    "dose_violation": bool(doses_in(p["advice"]) - ctx_doses), "retrieval_hit": (gold in [c["disease"] for c in rt["results"] if c["via"] == "direct"]) if (rt and gold) else None,
                    "review_ok": chk["ok"], "gen_s": g["total_s"], "n_in": g["n_in"], "n_out": g["n_out"], "raw_output": g["text"][:600]})
    return out, {"ret_ms_mean": ret_ms, "gen_s_mean": mean([o["gen_s"] for o in out]), "wall_s": wall, "peak_gpu_gb": peak_gpu(), "api_errors": sum(1 for g in gens if g.get("error"))}

def eval_graph(rows, name, pipe, variant):
    out, t0 = [], time.perf_counter()
    for r in rows:
        f = pipe.run(r["symptom_text"], variant); gold = r.get("diagnosis"); ans = f["status"] == "answered"
        srcs = f.get("sources", [])
        out.append({"id": r["id"], "question": r["symptom_text"][:300], "gold": gold, "gold_crop": r.get("crop"), "category": r.get("category"),
                    "status": f["status"], "fallback_type": f.get("type"), "pred": f.get("diagnosis"), "refused": not ans, "attempts": f["attempts"],
                    "correct": (ans and f.get("diagnosis") == gold) if gold else None, "crop_ok": (f.get("crop") == r.get("crop")) if (gold and ans) else (False if gold else None),
                    "advice": f.get("advice", ""), "citation_correct": (bool(srcs) and all(s["disease"] == gold for s in srcs)) if (gold and ans) else None,
                    "grounded": f.get("support"), "dose_violation": bool(doses_in(f.get("advice", ""))) and ans, "reasons": f.get("reasons"),
                    "gen_s": (f.get("gen_ms") or 0) / 1000, "gen_total_s": f["latency_ms"] / 1000, "n_in": 0, "n_out": 0, "retrieval_hit": None, "parse_ok": True, "pred_crop": f.get("crop")})
    return out, {"ret_ms_mean": None, "gen_s_mean": mean([o["gen_s"] for o in out]), "wall_s": time.perf_counter() - t0, "peak_gpu_gb": peak_gpu()}

def summarize(name, out, extra, classes, pipeline):
    m = {"suite": name, "n": len(out), **extra}
    if name == "ood":
        non = [o for o in out if o["category"] != "dose_unsupported"]; dose = [o for o in out if o["category"] == "dose_unsupported"]
        m.update({"refusal_rate": mean([o["refused"] for o in out]), "false_answer_rate": mean([not o["refused"] for o in non]),
                  "dose_hallucination_rate": mean([o["dose_violation"] for o in dose]), "dose_refusal_rate": mean([o["refused"] for o in dose])})
        for cat in sorted({o["category"] for o in out}): m[f"refusal_{cat}"] = mean([o["refused"] for o in out if o["category"] == cat])
        return m
    gold = [o["gold"] for o in out]; pred = [o["pred"] if o["pred"] in classes else None for o in out]
    c, per = clf_metrics(gold, pred, classes); m.update(c)
    m.update({"em_crop": mean([o["crop_ok"] for o in out]), "em_both": mean([bool(o["correct"] and o["crop_ok"]) for o in out]), "parse_ok_rate": mean([o["parse_ok"] for o in out]),
              "refusal_rate": mean([o["refused"] for o in out]), "wrong_answer_rate": mean([(not o["refused"]) and (not o["correct"]) for o in out]),
              "grounded_mean": mean([o["grounded"] for o in out]), "citation_correct_rate": mean([o["citation_correct"] for o in out]),
              "dose_violation_rate": mean([o["dose_violation"] for o in out]), "retrieval_hit_rate": mean([o["retrieval_hit"] for o in out])})
    if pipeline == "graph":
        m.update({"fallback_rate": m["refusal_rate"], "retry_rate": mean([o["attempts"] > 1 for o in out]), "false_accept_rate": m["wrong_answer_rate"],
                  "false_fallback_rate": m["refusal_rate"]})
    m["per_class_f1"] = json.dumps({k: round(v, 3) for k, v in per.items()})
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--rag", choices=["on", "off"], default="on")
    ap.add_argument("--pipeline", choices=["raw", "graph"], default="raw"); ap.add_argument("--suites", default="all")
    ap.add_argument("--limit", type=int); ap.add_argument("--batch_size", type=int)
    a = ap.parse_args(); cfg = load_config(); rag = a.rag == "on" or a.pipeline == "graph"
    run_id = f"{a.variant}__rag-{'on' if rag else 'off'}__{a.pipeline}"; rd = P(cfg["paths"]["results_dir"]) / "runs" / run_id; rd.mkdir(parents=True, exist_ok=True)
    R = get_retriever(cfg); classes = json.loads((P(cfg["paths"]["dataset_dir"]) / "classes.json").read_text())
    suites = list(SUITES) if a.suites == "all" else a.suites.split(",")
    bs = a.batch_size or cfg["generation"]["batch_size"]
    pipe = None
    if a.pipeline == "graph":
        from agri.orchestrator.graph import Pipeline; pipe = Pipeline(cfg, R)
    else:
        backend = resolve_backend(a.variant, cfg)
    rows_m = []
    for s in suites:
        rows = read_jsonl(P(cfg["paths"]["dataset_dir"]) / SUITES[s])[: a.limit]
        out, extra = eval_graph(rows, s, pipe, a.variant) if pipe else eval_raw(rows, s, R, backend, cfg, rag, bs)
        write_jsonl(rd / f"{s}_preds.jsonl", out)
        m = {"run_id": run_id, "variant": a.variant, "rag": rag, "pipeline": a.pipeline, **summarize(s, out, extra, classes, a.pipeline)}; rows_m.append(m)
        print(f"[{run_id}] {s}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float) and k in ("accuracy", "macro_f1", "rare_f1", "refusal_rate", "false_answer_rate")))
    csv = rd / "metrics.csv"
    df = pd.DataFrame(rows_m)
    if csv.exists():
        old = pd.read_csv(csv); old = old[~old["suite"].isin(df["suite"])]; df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv, index=False)
    (rd / "config.json").write_text(json.dumps({"variant": a.variant, "variant_cfg": load_variants()[a.variant], "rag": rag, "pipeline": a.pipeline, "kb_version": R.kb_version,
                                                "retrieval": cfg["retrieval"], "prompt": cfg["prompt"], "decision": cfg["decision"], "time": time.strftime("%F %T")}, indent=2, default=str))

if __name__ == "__main__":
    main()
