"""Swappable generation backends behind one interface:
   generate(messages, max_new_tokens) -> {text, n_in, n_out, ttft, tpot, total_s}
   generate_many(list_of_messages, max_new_tokens) -> list of the same (batched where possible)."""
import os, re, time
from concurrent.futures import ThreadPoolExecutor
from agri.config import P, load_variants, load_config
from agri import device as dev

class GenerationError(Exception):
    """Safe error: message never contains keys or internal details."""

class Backend:
    name = "base"
    def generate(self, messages, max_new_tokens=220): raise NotImplementedError
    def generate_many(self, batch, max_new_tokens=220, batch_size=8):
        return [self.generate(m, max_new_tokens) for m in batch]

# ------------------------------------------------------------------ local HF (FP16 / 8-bit / NF4 / LoRA / QLoRA)
class HFBackend(Backend):
    def __init__(self, v, name="hf"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch, self.name, self.v = torch, name, v
        path = str(P(v["model"])) if (P(v["model"])).exists() else v["model"]
        quant = v.get("quant", "none")
        self.dev = dev.get(); self.cuda = self.dev == "cuda"
        if quant != "none" and not self.cuda:
            raise GenerationError("8-bit/NF4 variants need an NVIDIA CUDA GPU (bitsandbytes); not available on Mac/CPU.")
        kw = {}
        if quant == "int8": kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float16 if self.cuda else torch.float32
        if self.cuda: kw["device_map"] = {"": 0}
        self.tok = AutoTokenizer.from_pretrained(path); self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, **kw)
        if v.get("adapter_dir"):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(P(v["adapter_dir"])))
        if not self.cuda: self.model.to(self.dev)      # MPS or CPU (CUDA is placed by device_map)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _text(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, messages, max_new_tokens=220):
        torch = self.torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        times = []
        class Tick(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                times.append(time.perf_counter()); return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        enc = self.tok(self._text(messages), return_tensors="pt").to(self.device)
        dev.sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id,
                                      stopping_criteria=StoppingCriteriaList([Tick()]))
        dev.sync()
        total = time.perf_counter() - t0
        new = out[0, enc["input_ids"].shape[1]:]
        n_out = int(new.shape[0])
        ttft = (times[0] - t0) if times else total
        tpot = ((times[-1] - times[0]) / (len(times) - 1)) if len(times) > 1 else 0.0
        return {"text": self.tok.decode(new, skip_special_tokens=True), "n_in": int(enc["input_ids"].shape[1]), "n_out": n_out,
                "ttft": ttft, "tpot": tpot, "total_s": total}

    def generate_many(self, batch, max_new_tokens=220, batch_size=8):
        torch, res = self.torch, []
        order = sorted(range(len(batch)), key=lambda i: len(str(batch[i])))
        outs = [None] * len(batch)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = self.tok([self._text(batch[i]) for i in idx], return_tensors="pt", padding=True).to(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                o = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
            dt = time.perf_counter() - t0
            L = enc["input_ids"].shape[1]
            for j, i in enumerate(idx):
                new = o[j, L:]
                n_out = int((new != self.tok.pad_token_id).sum())
                outs[i] = {"text": self.tok.decode(new, skip_special_tokens=True), "n_in": int(enc["attention_mask"][j].sum()),
                           "n_out": n_out, "ttft": None, "tpot": None, "total_s": dt / len(idx)}
        return outs

# ------------------------------------------------------------------ OpenAI-compatible API (Groq etc.)
class OpenAICompat(Backend):
    """Reads LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (optionally LLM_REASONING_EFFORT) from the environment."""
    def __init__(self, v=None, name="llm_api"):
        self.name = name
        self.key = os.environ.get("LLM_API_KEY", ""); self.base = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
        self.effort = os.environ.get("LLM_REASONING_EFFORT", "low" if "gpt-oss" in self.model else "")
        if not self.key or "your_" in self.key:
            raise GenerationError("LLM_API_KEY is not set. Put it in .env or Streamlit secrets.")

    def _call(self, messages, max_new_tokens):
        import requests
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max(max_new_tokens, 2048}
        if self.effort: body["reasoning_effort"] = self.effort
        for attempt in range(8):
            try:
                r = requests.post(self.base + "/chat/completions", json=body, timeout=90,
                                  headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            except requests.RequestException:
                time.sleep(min(2 * (attempt + 1), 20)); continue
            if r.status_code == 400 and "reasoning_effort" in r.text and "reasoning_effort" in body:
                body.pop("reasoning_effort"); continue
            if r.status_code in (429, 500, 502, 503):
                try: wait = float(r.headers.get("retry-after", 0))
                except ValueError: wait = 0
                time.sleep(min(max(wait, 2 * (attempt + 1)), 65)); continue
            if r.status_code != 200:
                raise GenerationError(f"LLM service returned status {r.status_code}")
            return r.json()
        raise GenerationError("LLM service unavailable or rate limited after retries")

    def generate(self, messages, max_new_tokens=220):
        t0 = time.perf_counter(); j = self._call(messages, max_new_tokens); total = time.perf_counter() - t0
        msg = j["choices"][0]["message"]; u = j.get("usage", {})
        return {"text": msg.get("content") or "", "n_in": u.get("prompt_tokens", 0), "n_out": u.get("completion_tokens", 0),
                "ttft": None, "tpot": None, "total_s": total}

    def generate_many(self, batch, max_new_tokens=220, batch_size=4):
        """LLM_MIN_INTERVAL_SEC>0 => sequential calls spaced by that many seconds (use on free tiers with tight rate limits)."""
        gap = float(os.environ.get("LLM_MIN_INTERVAL_SEC", "0") or 0)
        def one(m):
            try: return self.generate(m, max_new_tokens)
            except GenerationError as e:
                return {"text": "", "n_in": 0, "n_out": 0, "ttft": None, "tpot": None, "total_s": 0.0, "error": str(e)}
        if gap > 0:
            out = []
            for m in batch:
                t = time.perf_counter(); out.append(one(m)); time.sleep(max(0.0, gap - (time.perf_counter() - t)))
            return out
        with ThreadPoolExecutor(max_workers=min(batch_size, 4)) as ex:
            return list(ex.map(one, batch))

class Ollama(Backend):
    def __init__(self, v, name="ollama"):
        self.name, self.host, self.model = name, v.get("host", "http://127.0.0.1:11434").rstrip("/"), v["model"]
    def generate(self, messages, max_new_tokens=220):
        import requests
        try:
            r = requests.post(self.host + "/api/chat", timeout=300, json={"model": self.model, "messages": messages, "stream": False,
                              "options": {"temperature": 0, "num_predict": max_new_tokens}})
            r.raise_for_status(); j = r.json()
        except Exception:
            raise GenerationError("Ollama server not reachable.")
        ns = 1e9
        return {"text": j["message"]["content"], "n_in": j.get("prompt_eval_count", 0), "n_out": j.get("eval_count", 0),
                "ttft": (j.get("load_duration", 0) + j.get("prompt_eval_duration", 0)) / ns,
                "tpot": (j.get("eval_duration", 0) / max(1, j.get("eval_count", 1))) / ns, "total_s": j.get("total_duration", 0) / ns}

class Mock(Backend):
    """Deterministic stand-in for offline smoke tests (NOT a model): answers with the top retrieved disease."""
    name = "mock"
    def generate(self, messages, max_new_tokens=220):
        from agri.prompts.builder import answer_json, OOS_ADVICE
        u = messages[-1]["content"]; t0 = time.perf_counter()
        heads = re.findall(r"\[S(\d+)\] \(([^·]+) · ([^·]+) · ([^)]+)\)\n([^\n]*)", u)
        if not heads:
            txt = answer_json(None, "OUT_OF_SCOPE", OOS_ADVICE, [], True)
        else:
            _, crop, dis, _, _ = heads[0]
            cites = [f"S{n}" for n, c, d, s, _ in heads if d.strip() == dis.strip()]
            adv = " ".join(x[4] for x in heads if x[2].strip() == dis.strip() and x[3].strip() != "Symptoms")
            txt = answer_json(crop.strip(), dis.strip(), adv[:400], cites, False)
        return {"text": txt, "n_in": len(u.split()), "n_out": len(txt.split()), "ttft": 0.0, "tpot": 0.0, "total_s": time.perf_counter() - t0}

_CACHE = {}
def get_backend(variant):
    if variant in _CACHE: return _CACHE[variant]
    vs = load_variants()
    if variant not in vs: raise GenerationError(f"unknown variant {variant}")
    v = vs[variant]; kind = v["backend"]
    if _CACHE and kind == "hf":  # keep only one local model in memory
        for k in [k for k in _CACHE if load_variants()[k]["backend"] == "hf"]: _CACHE.pop(k)
        try:
            import gc; gc.collect(); dev.empty_cache()
        except Exception: pass
    b = {"hf": lambda: HFBackend(v, variant), "openai_compat": lambda: OpenAICompat(v, variant),
         "ollama": lambda: Ollama(v, variant), "mock": lambda: Mock()}[kind]()
    _CACHE[variant] = b
    return b
