"""Perplexity on held-out agri text (HF variants only). Fine-tuned models may drift on general agri text: this measures it.
python -m agri.evaluation.perplexity --variant fp16 [--file data/heldout_agri_text.txt | --hf_dataset KisanVaani/agriculture-qa-english-only --n 300]"""
import argparse, json, math
from agri.config import load_config, load_variants, P
from agri.generation.client import resolve_backend

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--file", default="data/heldout_agri_text.txt")
    ap.add_argument("--hf_dataset"); ap.add_argument("--n", type=int, default=300); a = ap.parse_args(); cfg = load_config()
    if load_variants()[a.variant]["backend"] != "hf": raise SystemExit("perplexity needs a local HF variant")
    import torch
    b = resolve_backend(a.variant, cfg)
    if a.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(a.hf_dataset, split="train"); col = "answers" if "answers" in ds.column_names else ds.column_names[-1]
        texts = [str(x) for x in ds[col][: a.n]]
    else:
        texts = [l.strip() for l in open(P(a.file), encoding="utf-8") if l.strip()]
    nll, ntok = 0.0, 0
    for t in texts:
        ids = b.tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(b.device)
        if ids.shape[1] < 2: continue
        with torch.no_grad(): loss = b.model(input_ids=ids, labels=ids).loss.item()
        nll += loss * (ids.shape[1] - 1); ntok += ids.shape[1] - 1
    res = {"variant": a.variant, "source": a.hf_dataset or a.file, "n_texts": len(texts), "n_tokens": ntok, "perplexity": math.exp(nll / ntok)}
    d = P(cfg["paths"]["results_dir"]) / "perplexity"; d.mkdir(parents=True, exist_ok=True); (d / f"{a.variant}.json").write_text(json.dumps(res, indent=2)); print(res)

if __name__ == "__main__":
    main()
