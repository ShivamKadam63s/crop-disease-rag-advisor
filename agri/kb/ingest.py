"""Document loading, cleaning, and parsing into disease cards."""
import hashlib, re, unicodedata
from collections import Counter
from pathlib import Path
from agri.config import ROOT

SECTIONS = ["symptoms", "favorable_conditions", "cultural_control", "biological_control", "chemical_control"]
SECTION_NAMES = {"overview": "Overview", "symptoms": "Symptoms", "favorable_conditions": "Favorable conditions",
                 "cultural_control": "Cultural control", "biological_control": "Biological control",
                 "chemical_control": "Chemical control", "mixed": "Mixed", "text": "Text"}

def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def clean_pages(pages):
    """For PDF/DOCX text: NFKC, de-hyphenate, drop page numbers and repeated headers/footers."""
    lines_per_page = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]
    cnt = Counter(l for ls in lines_per_page for l in set(ls))
    thresh = max(2, int(0.5 * len(pages)))
    out = []
    for ls in lines_per_page:
        keep = [l for l in ls if not (len(pages) > 2 and cnt[l] >= thresh and len(l) < 80) and not re.fullmatch(r"(page\s*)?\d{1,4}(\s*of\s*\d+)?", l, re.I)]
        t = unicodedata.normalize("NFKC", "\n".join(keep))
        t = re.sub(r"-\n(?=[a-z])", "", t)
        out.append(re.sub(r"[ \t]+", " ", t))
    return out

def load_documents(raw_dir):
    docs = []
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.suffix.lower() not in (".md", ".txt", ".pdf", ".docx"):
            continue
        fmt = p.suffix.lower().lstrip(".")
        if fmt in ("md", "txt"):
            text = p.read_text(encoding="utf-8")
            pages = [text]
        elif fmt == "pdf":
            from pypdf import PdfReader
            pages = clean_pages([(pg.extract_text() or "") for pg in PdfReader(str(p)).pages])
            text = "\n\n".join(pages)
        else:
            import docx
            pages = clean_pages(["\n".join(par.text for par in docx.Document(str(p)).paragraphs)])
            text = pages[0]
        try:
            rel = str(p.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(p)
        docs.append({"doc_id": p.stem, "path": rel, "format": fmt, "text": text, "pages": pages,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return docs

FIELD_RE = re.compile(r"^([a-z_]+):\s*(.*)$")

def is_structured(doc):
    return doc["format"] in ("md", "txt") and re.search(r"^## .+\n(?:.*\n)*?symptoms:", doc["text"], re.M) is not None

def parse_structured(doc):
    raw, pos = doc["text"], 0
    crop, meta, cur, cards = None, {}, None, []
    for line in raw.splitlines(keepends=True):
        s = line.strip()
        if line.startswith("## "):
            if cur: cards.append(cur)
            cur = {"disease": s[3:].strip(), "fields": {}}
        elif line.startswith("# "):
            crop = s[2:].strip()
        else:
            m = FIELD_RE.match(s)
            if m:
                k, v = m.groups()
                if cur is None:
                    meta[k] = v
                else:
                    off = line.index(":") + 1
                    off += len(line[off:]) - len(line[off:].lstrip())
                    cur["fields"][k] = {"text": v, "start": pos + off, "end": pos + off + len(v)}
        pos += len(line)
    if cur: cards.append(cur)
    out = []
    for c in cards:
        f = c["fields"]
        out.append({
            "card_id": f"{slug(crop)}-{slug(c['disease'])}", "crop": crop, "disease": c["disease"],
            "type": f.get("type", {}).get("text", "disease"),
            "aliases": [a.strip() for a in f.get("aliases", {}).get("text", "").split(";") if a.strip()],
            "pathogen": f.get("pathogen", {}).get("text", ""),
            "fields": {k: f[k] for k in SECTIONS if k in f},
            "doc_id": doc["doc_id"], "doc_path": doc["path"],
            "source_title": meta.get("source_title", doc["doc_id"]),
            "source_url": meta.get("source_url", ""), "license": meta.get("license", ""),
        })
    return out

def infer_crop(doc, glossary):
    name = (doc["doc_id"] + " " + doc["text"][:500]).lower()
    for crop, al in glossary["crops"].items():
        if any(re.search(rf"\b{re.escape(a)}\b", name) for a in al):
            return crop.capitalize()
    return "Unknown"
