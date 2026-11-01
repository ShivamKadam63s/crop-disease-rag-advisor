"""LangGraph flow: guard -> retrieve -> (gate) -> draft -> review -> accept | retry | fallback."""
import re, time
from typing import Any, TypedDict
from langgraph.graph import END, StateGraph
from agri.config import load_config
from agri.generation.backends import GenerationError
from agri.generation.client import resolve_backend
from agri.generation.parse import parse_output, REFUSALS
from agri.orchestrator.checks import review
from agri.orchestrator.fallback import make_fallback
from agri.orchestrator.intent import analyze_query
from agri.orchestrator.lang import detect_lang, localize, to_english
from agri.prompts.builder import MODES, build_messages
from agri.retrieval.client import get_retriever
from agri.textutils import DOSE_RE

REFUSAL_LIKE = re.compile(r"\s*(i['\u2019]?m sorry|sorry|i can['\u2019]?t|i cannot|as an ai)", re.I)

class S(TypedDict, total=False):
    question: str; variant: str; intent: str; attempt: int; strict: bool; k: int; query: str
    ret: Any; ctx: list; raw: str; parsed: dict; check: dict; final: dict; trace: list; gen_ms: float; err: str; tokens: dict
    mode: str; lang: str; q_en: str; unsupported: str; crop_in: str; other: list

class Pipeline:
    def __init__(self, cfg=None, retriever=None, backend_factory=None):
        self.cfg = cfg or load_config()
        self.R = retriever or get_retriever(self.cfg)
        self.labels = self.R.labels
        self.glossary = getattr(self.R, "glossary", None)
        if self.glossary is None:
            import json
            from agri.config import P
            self.glossary = json.loads(P(self.cfg["paths"]["glossary"]).read_text())
        self.backend_factory = backend_factory or (lambda v: resolve_backend(v, self.cfg))
        self.dc = self.cfg["decision"]
        self.app = self._compile()

    # ---- nodes ----
    def n_prep(self, s):
        """Hindi/Hinglish questions are translated to English before the guard and retrieval (the embedder is English-only)."""
        q, lang = s["question"], detect_lang(s["question"], self.glossary)
        tr = s["trace"] + [{"step": "prep", "detected_lang": lang, "mode": s["mode"]}]
        if lang != "en" and s["variant"] in self.TRANSLATORS:
            try:
                en = to_english(self.backend_factory(s["variant"]), q)
                if en and not REFUSAL_LIKE.match(en): q = en; tr[-1]["translated"] = en
            except Exception as e:   # translation is best effort: fall back to the original text
                tr[-1]["translate_error"] = type(e).__name__
        want = s["lang"] if s["lang"] != "auto" else lang
        return {"q_en": q, "query": q, "lang": want, "trace": tr}

    TRANSLATORS = ("llm_api", "ollama")

    def n_guard(self, s):
        q = s["q_en"]
        crop = self.R.detect_crop(q) if hasattr(self.R, "detect_crop") else None
        a = analyze_query(q, self.glossary, crop, [l["disease"] for l in self.labels])
        if s["question"] != q and analyze_query(s["question"], self.glossary)["intent"] == "dosage": a = {**a, "intent": "dosage"}   # translation can not hide a dose request
        intent = a["intent"]
        tr = s["trace"] + [{"step": "guard", "intent": intent, "crop": crop}]
        base = {"intent": intent, "trace": tr, "crop_in": crop}
        if intent in ("out_of_scope", "need_crop"): return {**base, "final": self._fb(intent, s)}
        if intent == "unsupported_crop": return {**base, "final": self._fb(intent, s, unsupported=a.get("unsupported"))}
        if intent == "dosage" and not a["crop_hit"]: return {**base, "final": self._fb("dose", s)}   # no crop named: never a dose
        return base

    def route_after_guard(self, s):
        if s.get("final"): return "end"
        return "info" if s["intent"] == "info" else "retrieve"

    def n_info(self, s):
        """'Which diseases are favoured by ...': answered extractively from the Favorable-conditions sections (no generation)."""
        crop = s.get("crop_in")
        ret = self.R.search("favorable conditions " + s["query"], k=12, parent_expand=False)
        rows = [c for c in ret["results"] if c["section"] == "favorable_conditions" and (not crop or c["crop"] == crop)][:4]
        tr = s["trace"] + [{"step": "info", "chunks": [c["chunk_id"] for c in rows]}]
        if not rows: return {"ret": ret, "trace": tr, "final": self._fb("insufficient", s, ret)}
        src = [self._src(f"S{i}", c) for i, c in enumerate(rows, 1)]
        adv = " | ".join(f"{c['disease']} [S{i}]: {c['raw_text'].strip().rstrip('.')}" for i, c in enumerate(rows, 1))
        return {"ret": ret, "trace": tr, "final": {"status": "answered", "kind": "info", "diagnosis": None, "crop": crop or rows[0]["crop"], "advice": adv,
                                                     "needs_expert": False, "sources": src, "support": 1.0, "confidence": {"level": "info"}}}

    def n_retrieve(self, s):
        ret = self.R.search(s["query"], k=s["k"])
        other = self._diseases(ret, s["query"]) if (s["attempt"] == 0 and (s["intent"] != "dosage")) else s.get("other")
        tr = s["trace"] + [{"step": "retrieve", "attempt": s["attempt"], "k": s["k"], "dense_top": round(ret["dense_top"], 3), "crop": ret["crop"],
                            "chunks": [(c["chunk_id"], c["via"]) for c in ret["results"]], "ms": round(ret["latency_ms"], 1)}]
        return {"ret": ret, "other": other, "trace": tr}

    def route_after_retrieve(self, s):
        weak = s["ret"]["dense_top"] < self.dc["min_dense_score"]
        has_dose = any(DOSE_RE.search(c["raw_text"]) for c in s["ret"]["results"])
        if s["intent"] == "dosage" and not has_dose: return "fallback_dose"
        if not s["ret"]["crop"] and not weak and self._crops_tied(s.get("other")): return "fallback_need_crop"
        if weak: return "retry" if s["attempt"] < self.dc["max_retries"] else "fallback_insufficient"
        return "draft"

    def n_draft(self, s):
        m = MODES[s["mode"]]
        msgs, ctx = build_messages(s["q_en"], self.labels, s["ret"]["results"], self.cfg["prompt"], strict=s["strict"], max_chunks=None, mode=s["mode"])
        variant = s["variant"]
        if s["attempt"] > 0 and self.dc.get("escalate_variant"): variant = self.dc["escalate_variant"]
        try:
            g = self.backend_factory(variant).generate(msgs, m["tokens"] or self.cfg["generation"]["max_new_tokens"])
        except GenerationError as e:
            return {"err": str(e), "final": self._fb("error", s), "trace": s["trace"] + [{"step": "draft", "error": str(e)}]}
        parsed = parse_output(g["text"], self.labels)
        return {"ctx": ctx, "raw": g["text"], "parsed": parsed, "gen_ms": g["total_s"] * 1000,
                "tokens": {"n_in": g["n_in"], "n_out": g["n_out"]},
                "trace": s["trace"] + [{"step": "draft", "variant": variant, "strict": s["strict"], "diagnosis": parsed["diagnosis"], "n_out": g["n_out"]}]}

    def route_after_draft(self, s):
        return "fallback_error" if s.get("err") else "review"

    def n_review(self, s):
        dc = {**self.dc, "min_support": self.dc["min_support"] * MODES[s["mode"]]["support"]}
        chk = review(s["parsed"], s["ctx"], self.labels, s["ret"]["dense_top"], dc)
        return {"check": chk, "trace": s["trace"] + [{"step": "review", "ok": chk["ok"], "reasons": chk["reasons"], "support": round(chk["support"], 2)}]}

    def route_after_review(self, s):
        c = s["check"]
        if c["ok"]: return "accept"
        if c["refusal"] == "OUT_OF_SCOPE": return "fallback_oos"
        if s["attempt"] < self.dc["max_retries"]: return "retry"
        return "fallback_dose" if "unsupported_dose" in c["reasons"] else "fallback_insufficient"

    def n_retry(self, s):
        q = self.R.expand_query(s["q_en"]) if hasattr(self.R, "expand_query") else s["q_en"]
        return {"attempt": s["attempt"] + 1, "strict": True, "k": min(s["k"], 3), "query": q, "trace": s["trace"] + [{"step": "retry", "attempt": s["attempt"] + 1}]}

    def _src(self, sid, c):
        return {"id": sid, "chunk_id": c["chunk_id"], "title": c["source_title"], "crop": c["crop"], "disease": c["disease"], "section": c["section_name"],
                "text": c["raw_text"], "doc_path": c["doc_path"], "page": c["page"], "start": c["start"], "end": c["end"], "url": c["source_url"]}

    def _diseases(self, ret, query):
        """Best dense score per (crop, disease) over a wider candidate list: used for the 'also consider' and cross-crop checks."""
        try: wide = self.R.search(query, k=10, parent_expand=False)["results"]
        except Exception: wide = ret["results"]
        best = {}
        for c in wide:
            if c.get("disease") and c.get("dense_score") is not None:
                k = (c["crop"], c["disease"]); best[k] = max(best.get(k, -1.0), c["dense_score"])
        return sorted(([k[0], k[1], round(v, 3)] for k, v in best.items()), key=lambda x: -x[2])

    def _crops_tied(self, other):
        """No crop in the question and the two best diseases belong to different crops with near-equal evidence."""
        if not other or len(other) < 2: return False
        top = other[0]; rival = next((o for o in other[1:] if o[0] != top[0]), None)
        return bool(rival) and top[2] - rival[2] < self.dc.get("ambiguity_margin", 0.03)

    def n_accept(self, s):
        p, ctx = s["parsed"], s["ctx"]
        ids = {f"S{i}": c for i, c in enumerate(ctx, 1)}
        src = [self._src(c, ids[c]) for c in p["citations"] if c in ids]
        gate, top = self.dc["min_dense_score"], s["ret"]["dense_top"]
        also = [f"{o[0]} - {o[1]}" for o in (s.get("other") or []) if o[1] != p["label"] and o[0] == p["crop"]
                and (s["other"][0][2] - o[2]) < self.dc.get("ambiguity_margin", 0.03)][:2]
        conf = {"dense_top": round(top, 3), "gate": gate, "level": "high" if top >= gate + 0.08 and not also else "medium", "support": round(s["check"]["support"], 2)}
        return {"final": {"status": "answered", "diagnosis": p["label"], "crop": p["crop"], "advice": p["advice"], "needs_expert": p["needs_expert"] or bool(also),
                          "sources": src, "support": s["check"]["support"], "also_consider": also, "confidence": conf}}

    def n_localize(self, s):
        f = dict(s["final"])
        if f.get("status") != "answered" or s["lang"] in ("en", "auto") or s["variant"] not in self.TRANSLATORS: return {}
        try: out, ok = localize(self.backend_factory(s["variant"]), f["advice"], s["lang"])
        except Exception: out, ok = f["advice"], False
        if ok and out != f["advice"]: f.update({"advice_en": f["advice"], "advice": out, "advice_lang": s["lang"]})
        return {"final": f, "trace": s["trace"] + [{"step": "localize", "lang": s["lang"], "ok": ok}]}

    def _fb(self, kind, s, ret=None, unsupported=None):
        chunks = (ret or s.get("ret") or {}).get("results") if (ret or s.get("ret")) else None
        return {"status": "fallback", **make_fallback(kind, self.cfg, self.labels, chunks, s["question"], crop=s.get("crop_in"), unsupported=unsupported)}

    def _mk_fb(self, kind):
        return lambda s: {"final": self._fb(kind, s), "trace": s["trace"] + [{"step": "fallback", "type": kind}]}

    def _compile(self):
        g = StateGraph(S)
        g.add_node("prep", self.n_prep); g.add_node("info", self.n_info); g.add_node("localize", self.n_localize); g.add_node("fallback_need_crop", self._mk_fb("need_crop"))
        g.add_node("guard", self.n_guard); g.add_node("retrieve", self.n_retrieve); g.add_node("draft", self.n_draft)
        g.add_node("review", self.n_review); g.add_node("retry", self.n_retry); g.add_node("accept", self.n_accept)
        g.add_node("fallback_oos", self._mk_fb("out_of_scope")); g.add_node("fallback_dose", self._mk_fb("dose"))
        g.add_node("fallback_insufficient", self._mk_fb("insufficient")); g.add_node("fallback_error", self._mk_fb("error"))
        g.set_entry_point("prep"); g.add_edge("prep", "guard")
        g.add_conditional_edges("guard", self.route_after_guard, {"end": END, "retrieve": "retrieve", "info": "info"})
        g.add_edge("info", END)
        g.add_conditional_edges("retrieve", self.route_after_retrieve, {"draft": "draft", "retry": "retry", "fallback_dose": "fallback_dose", "fallback_insufficient": "fallback_insufficient", "fallback_need_crop": "fallback_need_crop"})
        g.add_conditional_edges("draft", self.route_after_draft, {"review": "review", "fallback_error": "fallback_error"})
        g.add_conditional_edges("review", self.route_after_review, {"accept": "accept", "retry": "retry", "fallback_oos": "fallback_oos", "fallback_dose": "fallback_dose", "fallback_insufficient": "fallback_insufficient"})
        g.add_edge("retry", "retrieve")
        g.add_edge("accept", "localize"); g.add_edge("localize", END)
        for n in ("fallback_need_crop", "fallback_oos", "fallback_dose", "fallback_insufficient", "fallback_error"): g.add_edge(n, END)
        return g.compile()

    def run(self, question, variant, mode="quick", lang="auto"):
        t0 = time.perf_counter()
        out = self.app.invoke({"question": question, "variant": variant, "attempt": 0, "strict": False, "k": (MODES.get(mode) or MODES["quick"])["k"] or self.cfg["retrieval"]["top_k"],
                               "query": question, "trace": [], "mode": mode if mode in MODES else "quick", "lang": lang}, {"recursion_limit": 25})
        f = dict(out["final"]); f.update({"variant": variant, "attempts": out.get("attempt", 0) + 1, "trace": out["trace"],
                                          "kb_version": self.R.kb_version, "latency_ms": (time.perf_counter() - t0) * 1000,
                                          "gen_ms": out.get("gen_ms"), "intent": out.get("intent"), "mode": out.get("mode"), "lang": out.get("lang"),
                                          "raw_output": out.get("raw"), "reasons": (out.get("check") or {}).get("reasons", [])})
        return f
