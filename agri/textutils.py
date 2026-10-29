import re
STOP = set("the a an and or of to in on for with is are was were be been it its this that these those at by from as but not no if then so we you your i my our they their can may will should would could do does did have has had than into over under after before when while which what how much many any all some more most also only use used using per".split())
TOK = re.compile(r"[a-z0-9]+")
SENT = re.compile(r"(?<=[.;!?])\s+")
DOSE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?(?:\s*(?:-|–|to)\s*\d+(?:[.,]\d+)?)?)\s*"
    r"(%|percent\b|ppm\b|(?:ml|mg|kg|gms|gm|g|cc|l|lit|litres|litre|liters|liter|grams|gram)\b)"
    r"(?:\s*(?:/|per)\s*(?:l|lit|litre|liter|ha|hectare|acre|tank|plant)\b)?", re.I)

def tokenize(text, drop_stop=True):
    t = TOK.findall(text.lower())
    return [w for w in t if w not in STOP] if drop_stop else t

def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w

def content_words(text):
    return {stem(w) for w in tokenize(text) if len(w) > 3}

def jaccard(a, b):
    a, b = set(tokenize(a)), set(tokenize(b))
    return len(a & b) / max(1, len(a | b))

def norm_dose(s):
    s = s.lower().replace(" ", "").replace(",", ".")
    for a, b in (("litres", "l"), ("litre", "l"), ("liters", "l"), ("liter", "l"), ("lit", "l"), ("grams", "g"), ("gram", "g"), ("gms", "g"), ("gm", "g"), ("percent", "%"), ("per", "/"), ("to", "-"), ("–", "-")):
        s = s.replace(a, b)
    return s

def doses_in(text):
    return {norm_dose(m.group(0)) for m in DOSE_RE.finditer(text or "")}

def support_score(answer, context_texts):
    """Lexical support: fraction of the answer's content words found in the cited context."""
    aw = content_words(answer)
    if not aw:
        return 1.0
    cw = set()
    for t in context_texts:
        cw |= content_words(t)
    return len(aw & cw) / len(aw)

def words(text):
    return len(text.split())
