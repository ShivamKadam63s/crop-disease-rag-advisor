"""LoRA / QLoRA fine-tuning of Qwen2.5-0.5B-Instruct on the SFT data.
python -m agri.train.train --mode lora|qlora [--epochs 3 --lr 2e-4 --r 16 --bs 2 --accum 8 --max_len 1536]"""
import argparse, json, time
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from agri.config import load_config, P, read_jsonl

BASE = "Qwen/Qwen2.5-0.5B-Instruct"

class SFT(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items, self.dropped = [], 0
        for r in rows:
            p = tok(tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
            t = tok(r["target"] + tok.eos_token, add_special_tokens=False).input_ids
            if len(p) + len(t) > max_len: self.dropped += 1; continue
            self.items.append({"input_ids": p + t, "labels": [-100] * len(p) + t})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(pad):
    def f(b):
        m = max(len(x["input_ids"]) for x in b)
        ids = torch.tensor([x["input_ids"] + [pad] * (m - len(x["input_ids"])) for x in b])
        lab = torch.tensor([x["labels"] + [-100] * (m - len(x["labels"])) for x in b])
        att = torch.tensor([[1] * len(x["input_ids"]) + [0] * (m - len(x["input_ids"])) for x in b])
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lora", "qlora"], required=True); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--max_len", type=int, default=1536)
    a = ap.parse_args(); cfg = load_config()
    tok = AutoTokenizer.from_pretrained(BASE); cuda = torch.cuda.is_available()
    if not cuda: raise SystemExit("A CUDA GPU is required for training.")
    sd = P(cfg["paths"]["sft_dir"])
    tr, va = SFT(read_jsonl(sd / "train.jsonl"), tok, a.max_len), SFT(read_jsonl(sd / "val.jsonl"), tok, a.max_len)
    print(f"train {len(tr)} (dropped {tr.dropped}) val {len(va)}")
    bf16 = torch.cuda.get_device_capability()[0] >= 8   # Ampere+ only (T4 has no native bf16)
    if a.mode == "qlora":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16 if bf16 else torch.float32).cuda()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad); tot = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {tp:,} / {tot:,} = {100 * tp / tot:.2f}%")
    out_dir = P(cfg["paths"]["models_dir"]) / f"{a.mode}_adapter"
    args = TrainingArguments(output_dir=str(P("artifacts/tmp_train")), per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs, gradient_accumulation_steps=a.accum,
                             num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=5,
                             save_strategy="no", bf16=bf16, fp16=(not bf16), report_to="none", remove_unused_columns=False, optim="adamw_torch",
                             dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, data_collator=collate(tok.pad_token_id))
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    res = trainer.train(); secs = time.time() - t0
    model.save_pretrained(str(out_dir)); tok.save_pretrained(str(out_dir))      # save first: never lose the adapter to an eval OOM
    try: ev = trainer.evaluate()
    except Exception as e: print("val eval failed:", e); ev = {"eval_loss": float("nan")}
    rep = {"mode": a.mode, "train_seconds": secs, "train_loss": res.training_loss, "val_loss": ev["eval_loss"], "trainable_params": tp, "total_params": tot,
           "trainable_pct": 100 * tp / tot, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9, "epochs": a.epochs, "lr": a.lr, "r": a.r,
           "n_train": len(tr), "gpu": torch.cuda.get_device_name(0), "adapter_dir": str(out_dir)}
    r = P(cfg["paths"]["results_dir"]) / "training"; r.mkdir(parents=True, exist_ok=True)
    (r / f"{a.mode}.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
