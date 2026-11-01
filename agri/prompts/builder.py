"""Prompt construction. ONE source of truth used by inference, evaluation and SFT data building.
Baseline (no retrieval) and RAG prompts share rules and format so the comparison is fair."""
import json
from agri.textutils import jaccard, words

OOS_ADVICE = "I can only help with diseases and pests of the crops in my documents. Please contact your local agriculture officer."
INSUF_ADVICE = "My documents do not contain enough information to answer this reliably. Please consult your local agriculture officer."

def answer_json(crop, diagnosis, advice, citations, needs_expert):
    return json.dumps({"crop": crop, "diagnosis": diagnosis, "advice": advice, "citations": citations, "needs_expert": needs_expert}, ensure_ascii=False)

def labels_text(labels):
    by = {}
    for l in labels: by.setdefault(l["crop"], []).append(l["disease"])
    return "; ".join(f"{c}: {', '.join(d)}" for c, d in by.items())

MODES = {
    "quick": {"label": "Quick answer", "hint": "3-5 short lines", "tokens": None, "k": None, "support": 1.0, "directive": ""},
    "detailed": {"label": "Detailed guide", "hint": "step-by-step: confirm, act now, cultural, biological, chemical (label only), prevention", "tokens": 5096, "k": 5, "support": 0.8,
                 "directive": ("Write a DETAILED step-by-step guide inside \"advice\" as labeled sections, each on its own line, using the JSON newline escape `\\n` between sections. "
                               "Use these section labels when supported by the context: What to confirm in the field: ...; Do now: ...; Cultural control: ...; Biological control: ...; "
                               "Chemical control: ... (name only what the context names; NO dose, follow the product label); Prevention: .... "
                               "Omit unsupported sections. Do not use a pipe (|) as a separator. Keep the whole response valid one-line JSON.")},
    "simple": {"label": "Simple words", "hint": "very short sentences, numbered steps", "tokens": 800, "k": None, "support": 0.9,
               "directive": "Write \"advice\" in very simple words for a farmer with little schooling: 4-6 numbered steps (1. 2. 3.), each under 15 words, no technical terms."},
    "prevention": {"label": "Prevention plan", "hint": "what to do before and next season", "tokens": 2048, "k": 5, "support": 0.8,
                   "directive": "Write \"advice\" as a PREVENTION and monitoring plan for the next crop or season (seed, field hygiene, water and fertilizer, early signs to scout for), only from the context."},
    "organic": {"label": "Organic / low-chemical", "hint": "cultural + biological options only", "tokens": 2048, "k": 5, "support": 0.8,
                "directive": ("Write \"advice\" using ONLY cultural and biological control from the context. Do not recommend chemical sprays. "
                              "If the context has no biological option, say so and suggest consulting the agriculture officer.")},
}

def system_prompt(labels, rag, strict=False, mode="quick"):
    s = ("You are a crop disease advisory assistant for farmers in India. You only diagnose diseases and pests of the crops listed below.\n"
         f"Known crops and diseases/pests (use these exact names): {labels_text(labels)}.\n\n"
         "Rules:\n"
         "1. Choose \"diagnosis\" only from the known list. If the question is not about a disease or pest of these crops, set diagnosis to OUT_OF_SCOPE.\n"
         "2. Never invent pesticide names or doses. If you are unsure, set needs_expert to true.\n"
         "3. Reply with ONE JSON object and nothing else, with keys: crop, diagnosis, advice, citations, needs_expert.\n")
    if rag:
        s += ("4. Use ONLY the numbered context passages [S1], [S2], ... Put the ids of the passages you used in \"citations\", e.g. [\"S1\",\"S2\"].\n"
              "5. Copy any dose or quantity exactly as written in the context. If the context contains no dose, give no number.\n"
              "6. If the context is not enough to identify the problem, set diagnosis to INSUFFICIENT_CONTEXT.\n")
    else:
        s += "4. You have no reference documents, so \"citations\" must be an empty list [].\n"
    if mode != "quick" and rag and mode in MODES: s += "\nOUTPUT STYLE: " + MODES[mode]["directive"] + " Keep it valid JSON on one line (no raw line breaks inside strings).\n"
    if strict:
        s += ("\nYour previous answer was rejected by a checker. Be strict: keep advice " + ("to two short sentences " if mode == "quick" else "short ") + "copied from the cited passages, "
              "cite only passages that name the same disease you diagnose, and use INSUFFICIENT_CONTEXT if unsure.\n")
    return s

def _fmt_chunk(i, c):
    return f"[S{i}] ({c['crop']} · {c['disease'] or c['source_title']} · {c['section_name']})\n{c['raw_text']}"

def select_context(chunks, pcfg, max_chunks=None):
    """Deduplicate near-identical chunks, enforce a word budget, optionally reorder (best at both ends)."""
    kept, budget = [], pcfg["max_context_words"]
    for c in chunks:
        if any(jaccard(c["raw_text"], k["raw_text"]) > pcfg["dedup_jaccard"] for k in kept): continue
        if words(c["raw_text"]) > budget: continue
        budget -= words(c["raw_text"]); kept.append(c)
        if max_chunks and len(kept) >= max_chunks: break
    if pcfg.get("reorder"):
        d = [c for c in kept if c.get("via") == "direct"]; p = [c for c in kept if c.get("via") != "direct"]
        if len(d) >= 3: d = [d[0]] + d[2:] + [d[1]]
        kept = d + p
    return kept

def _fewshot(rag):
    q1 = "paddy leaves have many small oval brown spots with a grey middle and some spots on the husk of the grain"
    q2 = "my buffalo has stopped eating and has a high temperature"
    if rag:
        ctx1 = [{"crop": "Rice", "disease": "Brown spot", "section_name": "Symptoms", "raw_text": "small oval brown spots with grey centre and dark brown margin on leaves; similar dark spots on the grain husk"},
                {"crop": "Rice", "disease": "Brown spot", "section_name": "Cultural control", "raw_text": "Use treated healthy seed and apply balanced fertilizer including potash. Keep proper water in the field and avoid drought stress."}]
        ctx2 = [{"crop": "General guidance", "disease": None, "source_title": "Pesticide safety", "section_name": "Pesticide safety", "raw_text": "Use only pesticides registered for the crop and read the label before every spray."}]
        u1 = _user(q1, ctx1); u2 = _user(q2, ctx2)
        a1 = answer_json("Rice", "Brown spot", "Use treated healthy seed and apply balanced fertilizer including potash. Keep proper water in the field and avoid drought stress.", ["S1", "S2"], False)
    else:
        u1, u2 = _user(q1, None), _user(q2, None)
        a1 = answer_json("Rice", "Brown spot", "Use treated healthy seed, balanced fertilizer including potash and avoid water stress. Confirm with your local agriculture officer.", [], True)
    a2 = answer_json(None, "OUT_OF_SCOPE", OOS_ADVICE, [], True)
    return [{"role": "user", "content": u1}, {"role": "assistant", "content": a1}, {"role": "user", "content": u2}, {"role": "assistant", "content": a2}]

def _user(question, ctx):
    if ctx is None:
        return f"Farmer's question: {question}\n\nJSON answer:"
    block = "\n\n".join(_fmt_chunk(i, c) for i, c in enumerate(ctx, 1))
    return f"Context:\n{block}\n\nFarmer's question: {question}\n\nJSON answer:"

def build_messages(question, labels, chunks, pcfg, strict=False, max_chunks=None, mode="quick"):
    """chunks=None -> baseline (no retrieval). Returns (messages, displayed_context)."""
    rag = chunks is not None
    ctx = select_context(chunks, pcfg, max_chunks) if rag else None
    msgs = [{"role": "system", "content": system_prompt(labels, rag, strict, mode)}]
    if pcfg.get("fewshot"): msgs += _fewshot(rag)
    msgs.append({"role": "user", "content": _user(question, ctx)})
    return msgs, (ctx or [])
