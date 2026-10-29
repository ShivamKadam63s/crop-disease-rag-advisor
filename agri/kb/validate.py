"""KB validation. Errors block the build; warnings are reported for manual review."""
import json, re
from collections import Counter
from agri.config import ROOT
from agri.textutils import DOSE_RE

REQUIRED = ["symptoms", "cultural_control", "chemical_control"]

def validate(cards, chunks, docs, glossary, max_words):
    errors, warnings = [], []
    labels = [c for c in cards if c["type"] != "guidance"]
    names = Counter(c["disease"].lower() for c in labels)
    for n, c in names.items():
        if c > 1: errors.append(f"duplicate disease name across crops: {n}")
    for c in labels:
        for r in REQUIRED:
            if r not in c["fields"] or len(c["fields"][r]["text"]) < 20:
                errors.append(f"{c['card_id']}: missing/short field {r}")
        if len(c["fields"].get("symptoms", {}).get("text", "").split(";")) < 5:
            errors.append(f"{c['card_id']}: need >=5 ';'-separated symptom statements (used for dataset generation + train/test split)")
    seen = {}
    raw_by_doc = {d["doc_id"]: d["text"] for d in docs}
    for ch in chunks:
        if not ch["raw_text"].strip(): errors.append(f"{ch['chunk_id']}: empty chunk")
        if len(ch["raw_text"].split()) > max_words * 1.6: warnings.append(f"{ch['chunk_id']}: long chunk")
        for k in ("crop", "source_title", "doc_path", "chunk_id"):
            if not ch.get(k): errors.append(f"{ch['chunk_id']}: missing metadata {k}")
        if ch["text"] in seen: warnings.append(f"duplicate chunk text: {ch['chunk_id']} == {seen[ch['text']]}")
        seen[ch["text"]] = ch["chunk_id"]
        if ch.get("start") is not None:
            if raw_by_doc[ch["doc_id"]][ch["start"]:ch["end"]] != ch["raw_text"][: ch["end"] - ch["start"]]:
                errors.append(f"{ch['chunk_id']}: offsets do not match source text (citation highlight would be wrong)")
    chem = [ch["chunk_id"] for ch in chunks if DOSE_RE.search(ch["raw_text"])]
    if chem: warnings.append(f"{len(chem)} chunks contain dose-like strings; manually verify against source: {chem[:10]}")
    crops = {c["crop"].lower() for c in labels}
    for k in glossary["crops"]:
        if k not in crops: warnings.append(f"glossary crop '{k}' has no cards")
    dnames = {c["disease"].lower() for c in labels}
    for a, target in glossary["disease_aliases"].items():
        if target not in dnames and target not in {x.lower() for c in labels for x in c["aliases"]}:
            warnings.append(f"glossary alias '{a}' points to unknown disease '{target}'")
    report = {"n_docs": len(docs), "n_cards": len(cards), "n_labels": len(labels), "n_chunks": len(chunks),
              "errors": errors, "warnings": warnings}
    return report
