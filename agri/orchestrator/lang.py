"""Language handling around the (English-only) retrieval core: detect Hindi/Hinglish, translate the question to English before
retrieval, and rewrite the accepted answer in the farmer's language. Both steps use the same backend as the draft; they never
change what is retrieved or cited, and a translated answer that introduces a new number is discarded."""
import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LANGS = {"auto": "Same as my question", "en": "English", "hinglish": "Hinglish (Hindi in English letters)", "hi": "हिन्दी (Hindi)", "mr": "मराठी (Marathi)"}
_NAME = {"en": "English", "hinglish": "Hinglish: Hindi written in Roman/English letters, simple everyday words", "hi": "Hindi in Devanagari script", "mr": "Marathi in Devanagari script"}

def detect_lang(text, glossary):
    """Returns 'hi' (Devanagari), 'hinglish' (Roman-script Hindi) or 'en'."""
    if DEVANAGARI.search(text or ""): return "hi"
    toks = re.findall(r"[a-z]+", (text or "").lower())
    hits = sum(t in set(glossary.get("hinglish_words", [])) for t in toks)
    return "hinglish" if hits >= 2 else "en"

def _ask(backend, system, user, max_tokens=500):
    g = backend.generate([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens)
    return (g["text"] or "").strip().strip('"').strip()

def to_english(backend, text):
    """Faithful translation of a farmer's question; crop, disease and place names are kept."""
    return _ask(backend, "You translate farmers' questions about crop problems from Hindi, Hinglish or Marathi into plain English. "
                "Keep crop names, disease names and numbers. Output ONLY the English translation, nothing else.", text, 200)

_NUM = re.compile(r"\d+(?:[.,]\d+)?")

def localize(backend, advice, lang):
    """Rewrite advice in the target language. Returns (text, ok). Falls back to the English advice if the rewrite adds numbers."""
    if lang in ("en", "auto") or not advice: return advice, True
    out = _ask(backend, f"You rewrite crop-advice text in {_NAME[lang]}. Keep the meaning EXACTLY: add no new facts, no new products, "
               "no doses or numbers that are not in the original. Keep disease and chemical names recognisable (you may add them in English in brackets). "
               "Keep any numbering or | separators. Output ONLY the rewritten text.", advice, 900)
    if not out or not set(_NUM.findall(out)) <= set(_NUM.findall(advice)) | set("123456789"): return advice, False
    return out, True
