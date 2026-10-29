import re
DOSE_INTENT = re.compile(r"\b(how much|how many|dose|dosage|quantity|mix(?:ing)?|dilut\w*|ml|gm|gms|grams?|kg|litres?|liters?|per acre|per hectare|per tank|spray(?:ing)? (?:ratio|schedule|interval))\b", re.I)

def classify_intent(text, glossary, detected_crop=None):
    """Cheap first-line guard. Returns intent in {diagnose, dosage, out_of_scope}. Later layers (retrieval
    confidence, LLM refusal, review) catch what this misses."""
    t = text.lower()
    toks = set(re.findall(r"[a-z0-9\-]+", t))
    crop_hit = detected_crop is not None or any(a in t for al in glossary["crops"].values() for a in al)
    agri = len(toks & set(glossary["agri_terms"]))
    non = len(toks & set(glossary["non_agri_terms"]))
    if non and not crop_hit: return "out_of_scope"
    if agri == 0 and not crop_hit: return "out_of_scope"
    if DOSE_INTENT.search(t): return "dosage"
    return "diagnose"
