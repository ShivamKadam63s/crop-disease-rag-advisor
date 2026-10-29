import re
from agri.kb.ingest import SECTION_NAMES, SECTIONS
from agri.textutils import SENT, words

def split_words(text, max_words, overlap):
    """Sentence-aware recursive splitter with word overlap; hard-splits oversize sentences."""
    sents = [s for s in SENT.split(text.strip()) if s]
    chunks, cur, n = [], [], 0
    for s in sents:
        w = words(s)
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            keep, k = [], 0
            for t in reversed(cur):
                if k + words(t) > overlap: break
                keep.insert(0, t); k += words(t)
            cur, n = keep, k
        cur.append(s); n += w
    if cur: chunks.append(" ".join(cur))
    final = []
    for c in chunks:
        ws = c.split()
        if len(ws) <= int(max_words * 1.5):
            final.append(c)
        else:
            step = max(1, max_words - overlap)
            final += [" ".join(ws[i:i + max_words]) for i in range(0, len(ws), step)]
    return final

def _mk(card, section, i, header, piece, raw_piece, start, end, header_in_chunk):
    return {"chunk_id": f"{card['card_id']}::{section}::{i}", "crop": card["crop"], "disease": card["disease"],
            "card_id": card["card_id"], "type": card["type"], "section": section,
            "section_name": SECTION_NAMES[section], "doc_id": card["doc_id"], "doc_path": card["doc_path"],
            "source_title": card["source_title"], "source_url": card["source_url"], "page": 1,
            "start": start, "end": end, "raw_text": raw_piece,
            "text": f"{header}: {piece}" if (header_in_chunk and header) else piece}

def chunk_card_structured(card, max_words, overlap, header_in_chunk=True):
    out = []
    base = f"{card['crop']} > {card['disease']}"
    ov = f"{card['disease']} ({card['type']}) of {card['crop']}. Causal agent: {card['pathogen']}. Also known as: {', '.join(card['aliases']) or 'none'}."
    out.append(_mk(card, "overview", 0, f"{base} > Overview", ov, ov, None, None, header_in_chunk))
    for sec in SECTIONS:
        f = card["fields"].get(sec)
        if not f or f["text"].lower().startswith("not applicable"): continue
        for i, piece in enumerate(split_words(f["text"], max_words, overlap)):
            off = max(f["text"].find(piece[:30]), 0)
            s = f["start"] + off
            out.append(_mk(card, sec, i, f"{base} > {SECTION_NAMES[sec]}", piece, piece, s, min(s + len(piece), f["end"]), header_in_chunk))
    return out

def chunk_card_naive(card, max_words, overlap):
    body = " ".join(f"{SECTION_NAMES[k]}: {v['text']}" for k, v in card["fields"].items())
    text = f"{card['crop']} {card['disease']}. {body}"
    return [_mk(card, "mixed", i, "", p, p, None, None, False) for i, p in enumerate(split_words(text, max_words, overlap))]

def chunk_generic(doc, crop, max_words, overlap):
    out = []
    for pno, ptxt in enumerate(doc["pages"], 1):
        for i, piece in enumerate(split_words(re.sub(r"\n{2,}", " ", ptxt), max_words, overlap)):
            out.append({"chunk_id": f"{doc['doc_id']}::p{pno}::{i}", "crop": crop, "disease": None, "card_id": None,
                        "type": "document", "section": "text", "section_name": "Text", "doc_id": doc["doc_id"],
                        "doc_path": doc["path"], "source_title": doc["doc_id"], "source_url": "", "page": pno,
                        "start": None, "end": None, "raw_text": piece, "text": piece})
    return out
