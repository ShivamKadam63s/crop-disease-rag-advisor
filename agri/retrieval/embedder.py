import hashlib
import numpy as np
from agri.textutils import tokenize

class Embedder:
    """backend: st (sentence-transformers) | fastembed (ONNX, no torch; used for deployment) | hash (offline smoke test)."""
    def __init__(self, backend="auto", model="BAAI/bge-small-en-v1.5", query_prefix="", passage_prefix="", dim=384):
        self.model_name, self.qp, self.pp, self.dim = model, query_prefix, passage_prefix, dim
        self.backend = backend
        if backend == "auto":
            try:
                import sentence_transformers  # noqa
                self.backend = "st"
            except Exception:
                self.backend = "fastembed"
        if self.backend == "fastembed":
            try: import fastembed  # noqa
            except Exception: self.backend = "st"
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(model)
            self.dim = self.m.get_sentence_embedding_dimension()
        elif self.backend == "fastembed":
            from fastembed import TextEmbedding
            self.m = TextEmbedding(model_name=model)
            self.dim = len(next(iter(self.m.embed(["x"]))))
        elif self.backend != "hash":
            raise ValueError(f"unknown embedder backend {backend}")

    def _hash(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            toks = tokenize(t)
            for w in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return out

    def _enc(self, texts, prefix):
        texts = [prefix + t for t in texts]
        if self.backend == "st":
            v = self.m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        elif self.backend == "fastembed":
            v = np.array(list(self.m.embed(texts)))
        else:
            v = self._hash(texts)
        v = np.asarray(v, dtype="float32")
        n = np.linalg.norm(v, axis=1, keepdims=True); n[n == 0] = 1
        return v / n

    def encode_passages(self, texts): return self._enc(texts, self.pp)
    def encode_queries(self, texts): return self._enc(texts, self.qp)

def from_cfg(rcfg, model_override=None, backend_override=None):
    e = rcfg["embedder"]
    return Embedder(backend_override or e["backend"], model_override or e["model"], e.get("query_prefix", ""), e.get("passage_prefix", ""))
