"""Efficiency: disk size, GPU memory, TTFT, TPOT, throughput, energy, cost.
python -m agri.evaluation.efficiency --variant nf4 [--n 20 --batch 8]"""
import argparse, json, os, shutil, subprocess, tempfile, threading, time
import numpy as np
from agri.config import load_config, load_variants, P, read_jsonl
from agri.generation.client import resolve_backend
from agri.prompts.builder import build_messages
from agri.retrieval.client import get_retriever

def dir_size(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6

class PowerSampler:
    """Samples GPU power via nvidia-smi and integrates to energy (Wh). Returns None if unavailable."""
    def __init__(self, dt=0.2): self.dt, self.samples, self.stop, self.ok = dt, [], threading.Event(), True
    def _loop(self):
        while not self.stop.is_set():
            try:
                v = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, timeout=2).stdout.strip()
                self.samples.append(float(v.split()[0]))
            except Exception: self.ok = False; return
            time.sleep(self.dt)
    def __enter__(self): self.t0 = time.perf_counter(); self.th = threading.Thread(target=self._loop, daemon=True); self.th.start(); return self
    def __exit__(self, *a): self.stop.set(); self.th.join(); self.dur = time.perf_counter() - self.t0
    def wh(self): return float(np.mean(self.samples) * self.dur / 3600) if (self.ok and self.samples) else None

def disk_mb(backend, v):
    m = getattr(backend, "model", None)
    if m is None: return None
    tmp = tempfile.mkdtemp()
    try:
        base = m.base_model.model if hasattr(m, "peft_config") else m
        base.save_pretrained(tmp, safe_serialization=True); size = dir_size(tmp)
        if v.get("adapter_dir"): size += dir_size(str(P(v["adapter_dir"])))
        return size
    finally: shutil.rmtree(tmp, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rag", choices=["on", "off"], default="on"); a = ap.parse_args()
    cfg = load_config(); v = load_variants()[a.variant]; R = get_retriever(cfg); rows = read_jsonl(P(cfg["paths"]["dataset_dir"]) / "test.jsonl")[: a.n]
    msgs = [build_messages(r["symptom_text"], R.labels, R.search(r["symptom_text"])["results"] if a.rag == "on" else None, cfg["prompt"])[0] for r in rows]
    local = v["backend"] == "hf"
    try:
        import torch; cuda = torch.cuda.is_available()
    except Exception: torch, cuda = None, False
    if cuda: torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    backend = resolve_backend(a.variant, cfg)
    res = {"variant": a.variant, "rag": a.rag == "on", "gpu": torch.cuda.get_device_name(0) if cuda else None}
    if local and cuda:
        res["gpu_mem_after_load_gb"] = torch.cuda.memory_allocated() / 1e9
        try: res["disk_mb"] = disk_mb(backend, v)
        except Exception as e: res["disk_mb"] = None; res["disk_error"] = str(e)[:120]
        res["n_params_m"] = sum(p.numel() for p in backend.model.parameters()) / 1e6
    for m in msgs[:2]: backend.generate(m, cfg["generation"]["max_new_tokens"])      # warm-up
    if cuda: torch.cuda.reset_peak_memory_stats()
    ttft, tpot, tot, nout, nin = [], [], [], [], []
    with PowerSampler() as ps:
        for m in msgs:
            g = backend.generate(m, cfg["generation"]["max_new_tokens"])
            ttft.append(g["ttft"]); tpot.append(g["tpot"]); tot.append(g["total_s"]); nout.append(g["n_out"]); nin.append(g["n_in"])
    ok = lambda x: [i for i in x if i is not None]
    res.update({"n_prompts": len(msgs), "ttft_ms_p50": float(np.median(ok(ttft)) * 1000) if ok(ttft) else None, "ttft_ms_p95": float(np.percentile(ok(ttft), 95) * 1000) if ok(ttft) else None,
                "tpot_ms_mean": float(np.mean(ok(tpot)) * 1000) if ok(tpot) else None, "latency_s_mean": float(np.mean(tot)), "tokens_in_mean": float(np.mean(nin)), "tokens_out_mean": float(np.mean(nout)),
                "throughput_tok_s_single": float(sum(nout) / sum(tot)), "energy_wh_total": ps.wh() if local else None, "energy_wh_per_query": (ps.wh() / len(msgs)) if (local and ps.wh()) else None})
    if local and cuda:
        res["gpu_peak_gb_single"] = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); outs = backend.generate_many(msgs, cfg["generation"]["max_new_tokens"], a.batch); dt = time.perf_counter() - t0
        res[f"throughput_tok_s_batch{a.batch}"] = sum(o["n_out"] for o in outs) / dt; res[f"gpu_peak_gb_batch{a.batch}"] = torch.cuda.max_memory_allocated() / 1e9
    c = cfg["cost"]
    if local: res["cost_usd_per_1k_queries"] = c["gpu_usd_per_hour"] * (sum(tot) / 3600) / len(msgs) * 1000
    else: res["cost_usd_per_1k_queries"] = (np.mean(nin) * c["llm_usd_per_mtok_in"] + np.mean(nout) * c["llm_usd_per_mtok_out"]) / 1e6 * 1000
    d = P(cfg["paths"]["results_dir"]) / "efficiency"; d.mkdir(parents=True, exist_ok=True); (d / f"{a.variant}__rag-{a.rag}.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
