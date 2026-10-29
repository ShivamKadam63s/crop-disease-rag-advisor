import json, re

REFUSALS = {"OUT_OF_SCOPE", "INSUFFICIENT_CONTEXT"}

def _repair(s):
    s = s.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def normalize_label(name, labels):
    """Map a predicted diagnosis string onto the known label list (exact, alias, or substring)."""
    if not name or not isinstance(name, str): return None
    n = name.strip().lower()
    if n.upper().replace(" ", "_") in REFUSALS: return n.upper().replace(" ", "_")
    for l in labels:
        if n == l["disease"].lower(): return l["disease"]
    for l in labels:
        if n in [a.lower() for a in l["aliases"]]: return l["disease"]
    for l in labels:
        if l["disease"].lower() in n or n in l["disease"].lower(): return l["disease"]
    return None

def parse_output(text, labels):
    """Returns dict(parse_ok, crop, diagnosis(raw), label(normalized or None), advice, citations, needs_expert)."""
    out = {"parse_ok": False, "crop": None, "diagnosis": None, "label": None, "advice": "", "citations": [], "needs_expert": True}
    text = text or ""
    m = re.search(r"\{.*\}", text, re.S)
    obj = None
    if m:
        for cand in (m.group(0), _repair(m.group(0)), _repair(m.group(0)).replace("'", '"')):
            try: obj = json.loads(cand); break
            except Exception: continue
    if isinstance(obj, dict):
        out["parse_ok"] = True
        out["crop"] = obj.get("crop"); out["diagnosis"] = obj.get("diagnosis")
        out["advice"] = str(obj.get("advice") or "")
        c = obj.get("citations") or []
        out["citations"] = [str(x).strip("[]") for x in c] if isinstance(c, list) else []
        out["needs_expert"] = bool(obj.get("needs_expert", True))
    else:
        md = re.search(r'"?diagnosis"?\s*[:=]\s*"?([^",}\n]+)', text)
        if md: out["diagnosis"] = md.group(1).strip()
        mc = re.search(r'"?crop"?\s*[:=]\s*"?([^",}\n]+)', text)
        if mc: out["crop"] = mc.group(1).strip()
    out["label"] = normalize_label(out["diagnosis"], labels)
    if isinstance(out["crop"], str): out["crop"] = out["crop"].strip().capitalize()
    return out
