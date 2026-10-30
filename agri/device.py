"""Hardware detection so the same code runs on NVIDIA (CUDA), Apple Silicon (MPS) and plain CPU.
get() -> 'cuda' | 'mps' | 'cpu'. bitsandbytes (8-bit / NF4 / QLoRA) only works on CUDA; everything else works on all three."""
import platform, subprocess

def get():
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    except Exception: pass
    return "cpu"

def name():
    d = get()
    if d == "cuda":
        import torch; return torch.cuda.get_device_name(0)
    if d == "mps":
        try: chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception: chip = ""
        return f"Apple MPS ({chip or platform.machine()})"
    return None

def sync():
    d = get()
    if d == "cuda": import torch; torch.cuda.synchronize()
    elif d == "mps": import torch; torch.mps.synchronize()

def empty_cache():
    d = get()
    if d == "cuda": import torch; torch.cuda.empty_cache()
    elif d == "mps": import torch; torch.mps.empty_cache()

def reset_peak():
    if get() == "cuda": import torch; torch.cuda.reset_peak_memory_stats()   # MPS has no peak counter

def alloc_gb():
    d = get()
    if d == "cuda": import torch; return torch.cuda.memory_allocated() / 1e9
    if d == "mps": import torch; return torch.mps.current_allocated_memory() / 1e9
    return None

def peak_gb():
    """CUDA: true peak. MPS: driver-allocated memory at call time (an upper-bound proxy, since MPS exposes no peak counter)."""
    d = get()
    if d == "cuda": import torch; return torch.cuda.max_memory_allocated() / 1e9
    if d == "mps": import torch; return torch.mps.driver_allocated_memory() / 1e9
    return None
