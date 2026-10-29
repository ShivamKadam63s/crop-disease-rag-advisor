"""Retriever: hybrid (dense + BM25 with reciprocal rank fusion), crop filter, alias expansion,
long-query windowing, optional cross-encoder rerank and parent-document expansion.
Independent of generation: used in-process or behind the FastAPI service."""
import json, re, time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from agri.config import load_config, P, read_jsonl
from agri.retrieval.embedder import from_cfg
from agri.retrieval.store import VectorStore
from agri.textutils import tokenize

MGMT = ["cultural_control", "biological_control", "chemical_control"]

class Retriever:
    def __init__(self, kb_dir=None, cfg=None, embedder=None):
        self.cfg = cfg or load_config()
        self.r = self.cfg["retrieval"]
        self.dir = Path(kb_dir or P(self.cfg["paths"]["kb_dir"]))
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.kb_version = self.manifest["kb_version"]
        self.chunks = read_jsonl(self.dir / "chunks.jsonl")
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.glossary = json.loads(P(self.cfg["paths"]["glossary"]).read_text())
        self.labels = json.loads((self.dir / "labels.json").read_text())
        self.store = VectorStore.load(str(self.dir), self.manifest.get("index_type", "flat"))
        be = self.manifest["embedder_backend"]
        try:
            self.embedder = embedder or from_cfg(self.r, self.manifest["embedder_model"], be)
        except Exception:   # e.g. index built with sentence-transformers but this host only has fastembed
            self.embedder = embedder or from_cfg(self.r, self.manifest["embedder_model"], "fastembed")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._rr = None

    # ---- query understanding -------------------------------------------------
    def detect_crop(self, q):
        ql = q.lower(); hit = set()
        for crop, al in self.glossary["crops"].items():
            if any(re.search(rf"\b{re.escape(a)}\b", ql) for a in al): hit.add(crop.capitalize())
        return next(iter(hit)) if len(hit) == 1 else None

    def expand_query(self, q):
        ql, extra = q.lower(), []
        for a, t in {**self.glossary["disease_aliases"], **self.glossary["chemical_aliases"]}.items():
            if re.search(rf"\b{re.escape(a)}\b", ql) and t not in ql: extra.append(t)
        return q + (" " + " ".join(extra) if extra else "")

    # ---- ranking ---------------------------------------------------------------
    def _rank_one(self, q, mask, mode, n_cand):
        dense_s, dense_i, bm_rank = {}, [], []
        if mode in ("dense", "hybrid"):
            qv = self.embedder.encode_queries([q])
            s, i = self.store.search(qv, min(len(self.chunks), max(n_cand * 4, 60)))
            for sc, ix in zip(s[0], i[0]):
                dense_s[int(ix)] = float(sc)
            dense_i = [int(ix) for ix in i[0] if mask[int(ix)]][:n_cand]
        if mode in ("bm25", "hybrid"):
            sc = self.bm25.get_scores(tokenize(q))
            order = [int(ix) for ix in np.argsort(-sc) if mask[int(ix)] and sc[int(ix)] > 0][:n_cand]
            bm_rank = order
        fused = {}
        for lst in (dense_i, bm_rank):
            for r, ix in enumerate(lst):
                fused[ix] = fused.get(ix, 0.0) + 1.0 / (self.r["rrf_k"] + r + 1)
        return fused, dense_s

    def search(self, query, k=None, crop="auto", mode=None, rerank=None, expand=True, parent_expand=None):
        t0 = time.perf_counter()
        k = k or self.r["top_k"]
        mode = mode or ("hybrid" if self.r["hybrid"] else "dense")
        q = self.expand_query(query) if expand else query
        if crop == "auto":
            crop = self.detect_crop(query) if self.r["crop_filter"] else None
        mask = np.ones(len(self.chunks), dtype=bool)
        if crop:
            m2 = np.array([c["crop"] == crop or c["type"] == "guidance" for c in self.chunks])
            if m2.sum() >= k: mask = m2
            else: crop = None
        n_cand = self.r["candidates"]
        lq = self.r["long_query"]
        wl = q.split()
        if lq["enabled"] and len(wl) > lq["trigger_words"]:
            fused, dense_s = {}, {}
            for st in range(0, max(1, len(wl) - lq["window_words"] + lq["stride_words"]), lq["stride_words"]):
                f, d = self._rank_one(" ".join(wl[st:st + lq["window_words"]]), mask, mode, n_cand)
                for ix, v in f.items(): fused[ix] = max(fused.get(ix, 0.0), v)
                for ix, v in d.items(): dense_s[ix] = max(dense_s.get(ix, -1.0), v)
        else:
            fused, dense_s = self._rank_one(q, mask, mode, n_cand)
        ranked = sorted(fused, key=lambda i: -fused[i])
        use_rr = self.r["reranker"]["enabled"] if rerank is None else rerank
        if use_rr and ranked:
            ranked = self._rerank(q, ranked[:n_cand])
        top = ranked[:k]
        dense_top = max((dense_s[i] for i in dense_s if mask[i]), default=0.0) if dense_s else 0.0
        results = [self._fmt(i, fused.get(i, 0.0), dense_s.get(i), "direct") for i in top]
        pe = self.r["parent_expand"] if parent_expand is None else parent_expand
        if pe and results:
            results += self._parents(results)
        return {"query": query, "crop": crop, "results": results, "dense_top": dense_top,
                "kb_version": self.kb_version, "latency_ms": (time.perf_counter() - t0) * 1000}

    def _fmt(self, i, score, dense, via):
        c = self.chunks[i]
        return {**{k: c[k] for k in ("chunk_id", "text", "raw_text", "crop", "disease", "section", "section_name", "type",
                                     "doc_id", "doc_path", "source_title", "source_url", "page", "start", "end")},
                "score": float(score), "dense_score": dense, "via": via}

    def _parents(self, results):
        """Parent-document expansion: add management chunks of the highest-scoring disease."""
        sc = {}
        for r in results:
            if r["disease"]: sc[(r["crop"], r["disease"])] = sc.get((r["crop"], r["disease"]), 0) + r["score"]
        if not sc: return []
        crop, dis = max(sc, key=sc.get)
        have = {r["chunk_id"] for r in results}
        add = [self._fmt(i, 0.0, None, "parent") for i, c in enumerate(self.chunks)
               if c["crop"] == crop and c["disease"] == dis and c["section"] in MGMT and c["chunk_id"] not in have]
        return add

    def _rerank(self, q, idxs):
        if self._rr is None:
            from sentence_transformers import CrossEncoder
            self._rr = CrossEncoder(self.r["reranker"]["model"])
        s = self._rr.predict([(q, self.chunks[i]["text"]) for i in idxs])
        return [i for _, i in sorted(zip(-np.asarray(s), idxs))]

    def get_chunk(self, chunk_id):
        i = self.by_id.get(chunk_id)
        return None if i is None else self._fmt(i, 0.0, None, "direct")
