"""LangGraph flow: guard -> retrieve -> (gate) -> draft -> review -> accept | retry | fallback."""
import time
from typing import Any, TypedDict
from langgraph.graph import END, StateGraph
from agri.config import load_config
from agri.generation.backends import GenerationError
from agri.generation.client import resolve_backend
from agri.generation.parse import parse_output, REFUSALS
from agri.orchestrator.checks import review
from agri.orchestrator.fallback import make_fallback
from agri.orchestrator.intent import classify_intent
from agri.prompts.builder import build_messages
from agri.retrieval.client import get_retriever
from agri.textutils import DOSE_RE

class S(TypedDict, total=False):
    question: str; variant: str; intent: str; attempt: int; strict: bool; k: int; query: str
    ret: Any; ctx: list; raw: str; parsed: dict; check: dict; final: dict; trace: list; gen_ms: float; err: str; tokens: dict

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
    def n_guard(self, s):
        intent = classify_intent(s["question"], self.glossary, self.R.detect_crop(s["question"]) if hasattr(self.R, "detect_crop") else None)
        tr = s["trace"] + [{"step": "guard", "intent": intent}]
        if intent == "out_of_scope":
            return {"intent": intent, "trace": tr, "final": self._fb("out_of_scope", s)}
        return {"intent": intent, "trace": tr}

    def n_retrieve(self, s):
        ret = self.R.search(s["query"], k=s["k"])
        tr = s["trace"] + [{"step": "retrieve", "attempt": s["attempt"], "k": s["k"], "dense_top": round(ret["dense_top"], 3), "crop": ret["crop"],
                            "chunks": [(c["chunk_id"], c["via"]) for c in ret["results"]], "ms": round(ret["latency_ms"], 1)}]
        return {"ret": ret, "trace": tr}

    def route_after_retrieve(self, s):
        weak = s["ret"]["dense_top"] < self.dc["min_dense_score"]
        has_dose = any(DOSE_RE.search(c["raw_text"]) for c in s["ret"]["results"])
        if s["intent"] == "dosage" and not has_dose: return "fallback_dose"
        if weak: return "retry" if s["attempt"] < self.dc["max_retries"] else "fallback_insufficient"
        return "draft"

    def n_draft(self, s):
        msgs, ctx = build_messages(s["question"], self.labels, s["ret"]["results"], self.cfg["prompt"], strict=s["strict"], max_chunks=None)
        variant = s["variant"]
        if s["attempt"] > 0 and self.dc.get("escalate_variant"): variant = self.dc["escalate_variant"]
        try:
            g = self.backend_factory(variant).generate(msgs, self.cfg["generation"]["max_new_tokens"])
        except GenerationError as e:
            return {"err": str(e), "final": self._fb("error", s), "trace": s["trace"] + [{"step": "draft", "error": str(e)}]}
        parsed = parse_output(g["text"], self.labels)
        return {"ctx": ctx, "raw": g["text"], "parsed": parsed, "gen_ms": g["total_s"] * 1000,
                "tokens": {"n_in": g["n_in"], "n_out": g["n_out"]},
                "trace": s["trace"] + [{"step": "draft", "variant": variant, "strict": s["strict"], "diagnosis": parsed["diagnosis"], "n_out": g["n_out"]}]}

    def route_after_draft(self, s):
        return "fallback_error" if s.get("err") else "review"

    def n_review(self, s):
        chk = review(s["parsed"], s["ctx"], self.labels, s["ret"]["dense_top"], self.dc)
        return {"check": chk, "trace": s["trace"] + [{"step": "review", "ok": chk["ok"], "reasons": chk["reasons"], "support": round(chk["support"], 2)}]}

    def route_after_review(self, s):
        c = s["check"]
        if c["ok"]: return "accept"
        if c["refusal"] == "OUT_OF_SCOPE": return "fallback_oos"
        if s["attempt"] < self.dc["max_retries"]: return "retry"
        return "fallback_dose" if "unsupported_dose" in c["reasons"] else "fallback_insufficient"

    def n_retry(self, s):
        q = self.R.expand_query(s["question"]) if hasattr(self.R, "expand_query") else s["question"]
        return {"attempt": s["attempt"] + 1, "strict": True, "k": min(s["k"], 3), "query": q, "trace": s["trace"] + [{"step": "retry", "attempt": s["attempt"] + 1}]}

    def n_accept(self, s):
        p, ctx = s["parsed"], s["ctx"]
        ids = {f"S{i}": c for i, c in enumerate(ctx, 1)}
        src = [{"id": c, "chunk_id": ids[c]["chunk_id"], "title": ids[c]["source_title"], "crop": ids[c]["crop"], "disease": ids[c]["disease"],
                "section": ids[c]["section_name"], "text": ids[c]["raw_text"], "doc_path": ids[c]["doc_path"], "page": ids[c]["page"],
                "start": ids[c]["start"], "end": ids[c]["end"], "url": ids[c]["source_url"]} for c in p["citations"] if c in ids]
        return {"final": {"status": "answered", "diagnosis": p["label"], "crop": p["crop"], "advice": p["advice"], "needs_expert": p["needs_expert"],
                          "sources": src, "support": s["check"]["support"]}}

    def _fb(self, kind, s):
        chunks = s["ret"]["results"] if s.get("ret") else None
        return {"status": "fallback", **make_fallback(kind, self.cfg, self.labels, chunks, s["question"])}

    def _mk_fb(self, kind):
        return lambda s: {"final": self._fb(kind, s), "trace": s["trace"] + [{"step": "fallback", "type": kind}]}

    def _compile(self):
        g = StateGraph(S)
        g.add_node("guard", self.n_guard); g.add_node("retrieve", self.n_retrieve); g.add_node("draft", self.n_draft)
        g.add_node("review", self.n_review); g.add_node("retry", self.n_retry); g.add_node("accept", self.n_accept)
        g.add_node("fallback_oos", self._mk_fb("out_of_scope")); g.add_node("fallback_dose", self._mk_fb("dose"))
        g.add_node("fallback_insufficient", self._mk_fb("insufficient")); g.add_node("fallback_error", self._mk_fb("error"))
        g.set_entry_point("guard")
        g.add_conditional_edges("guard", lambda s: "end" if s.get("final") else "retrieve", {"end": END, "retrieve": "retrieve"})
        g.add_conditional_edges("retrieve", self.route_after_retrieve, {"draft": "draft", "retry": "retry", "fallback_dose": "fallback_dose", "fallback_insufficient": "fallback_insufficient"})
        g.add_conditional_edges("draft", self.route_after_draft, {"review": "review", "fallback_error": "fallback_error"})
        g.add_conditional_edges("review", self.route_after_review, {"accept": "accept", "retry": "retry", "fallback_oos": "fallback_oos", "fallback_dose": "fallback_dose", "fallback_insufficient": "fallback_insufficient"})
        g.add_edge("retry", "retrieve")
        for n in ("accept", "fallback_oos", "fallback_dose", "fallback_insufficient", "fallback_error"): g.add_edge(n, END)
        return g.compile()

    def run(self, question, variant):
        t0 = time.perf_counter()
        out = self.app.invoke({"question": question, "variant": variant, "attempt": 0, "strict": False, "k": self.cfg["retrieval"]["top_k"],
                               "query": question, "trace": []}, {"recursion_limit": 25})
        f = dict(out["final"]); f.update({"variant": variant, "attempts": out.get("attempt", 0) + 1, "trace": out["trace"],
                                          "kb_version": self.R.kb_version, "latency_ms": (time.perf_counter() - t0) * 1000,
                                          "gen_ms": out.get("gen_ms"), "intent": out.get("intent"),
                                          "raw_output": out.get("raw"), "reasons": (out.get("check") or {}).get("reasons", [])})
        return f
