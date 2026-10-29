"""Merge the LoRA adapter into FP16 base weights -> artifacts/models/lora_merged  (variant: lora_fp16, lora_merged_nf4)"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from agri.config import load_config, P
from agri.train.train import BASE

def main():
    cfg = load_config(); md = P(cfg["paths"]["models_dir"]); out = md / "lora_merged"
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)   # merge in fp32 on CPU (safe), then cast to fp16
    m = PeftModel.from_pretrained(m, str(md / "lora_adapter")).merge_and_unload().half()
    m.save_pretrained(str(out), safe_serialization=True); AutoTokenizer.from_pretrained(BASE).save_pretrained(str(out)); print("merged ->", out)

if __name__ == "__main__":
    main()
