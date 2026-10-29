"""Config + env loading. Keys come ONLY from environment / .env / Streamlit secrets."""
import json, os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def _load_dotenv():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_dotenv()

def load_config():
    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    cal = ROOT / "artifacts/calibration.json"
    if cal.exists():
        cfg["decision"].update(json.loads(cal.read_text()))
    if os.environ.get("AGRI_KB_DIR"):
        cfg["paths"]["kb_dir"] = os.environ["AGRI_KB_DIR"]
    return cfg

def load_variants():
    return yaml.safe_load((ROOT / "configs/variants.yaml").read_text())

def P(rel):
    return Path(rel) if os.path.isabs(str(rel)) else ROOT / rel

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
