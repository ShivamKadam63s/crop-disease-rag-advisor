import re
DOSE_INTENT = re.compile(r"\b(how much|how many|dose|dosage|quantity|mix(?:ing)?|dilut\w*|ml|gm|gms|grams?|kg|litres?|liters?|per acre|per hectare|per tank|spray(?:ing)? (?:ratio|schedule|interval))\b", re.I)

def _toks(t): return set(re.findall(r"[a-z0-9\-]+", t))

def _has_word(t, words):
    return next((w for w in words if re.search(rf"\b{re.escape(w)}\b", t)), None)

def analyze_query(text, glossary, detected_crop=None, disease_names=()):
    """Cheap first-line guard. Returns dict(intent, ...). intent in
    {diagnose, dosage, info, need_crop, unsupported_crop, out_of_scope}. Later layers (retrieval confidence, LLM refusal,
    review) catch what this misses."""
    t = text.lower(); toks = _toks(t)
    crop_hit = detected_crop is not None or any(re.search(rf"\b{re.escape(a)}\b", t) for al in glossary["crops"].values() for a in al)
    agri = len(toks & set(glossary["agri_terms"]))
    non = len(toks & set(glossary["non_agri_terms"]))
    out = {"intent": "diagnose", "crop_hit": bool(crop_hit)}
    # animal / human-health words win even when a crop word is present ("my rice spot has fever and is not eating")
    if toks & set(glossary.get("strong_non_agri_terms", [])): return {**out, "intent": "out_of_scope"}
    unsup = _has_word(t, glossary.get("unsupported_crops", []))
    if unsup and not crop_hit: return {**out, "intent": "unsupported_crop", "unsupported": unsup}
    # dose questions are recognised BEFORE the crop requirement: no crop named is still a dose question
    chem = bool(toks & set(glossary.get("chem_terms", [])) | toks & set(glossary.get("chemical_aliases", {})))
    if DOSE_INTENT.search(t) and (crop_hit or agri or chem): return {**out, "intent": "dosage"}
    if non and not crop_hit: return {**out, "intent": "out_of_scope"}
    if agri == 0 and not crop_hit: return {**out, "intent": "out_of_scope"}
    if any(re.search(p, t) for p in glossary.get("info_patterns", [])) and not (toks & set(glossary.get("symptom_terms", [])) - {"humid"}) :
        return {**out, "intent": "info" if crop_hit else "need_crop"}
    if not crop_hit:
        named = any(d.lower() in t for d in disease_names)
        spec = len(toks & set(glossary.get("symptom_terms", []))) + 2 * len(toks & set(glossary.get("crop_hint_terms", [])))
        if not named and spec < 2: return {**out, "intent": "need_crop"}
    return out

def classify_intent(text, glossary, detected_crop=None):
    return analyze_query(text, glossary, detected_crop)["intent"]
