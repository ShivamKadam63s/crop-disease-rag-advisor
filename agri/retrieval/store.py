"""Vector index behind a small interface. FAISS (flat or HNSW) with a numpy fallback."""
import numpy as np
try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

class VectorStore:
    def __init__(self, dim, index_type="flat"):
        self.dim, self.index_type, self.vecs, self.index = dim, index_type, None, None

    def add(self, vecs):
        vecs = np.ascontiguousarray(vecs, dtype="float32")
        self.vecs = vecs
        if faiss is None:
            return
        if self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 128
            self.index.hnsw.efSearch = 64
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

    def search(self, q, k):
        k = min(k, len(self.vecs))
        if self.index is not None:
            s, i = self.index.search(np.ascontiguousarray(q, dtype="float32"), k)
            return s, i
        sims = q @ self.vecs.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, 1), idx

    def save(self, directory):
        np.save(f"{directory}/embeddings.npy", self.vecs)
        if self.index is not None:
            faiss.write_index(self.index, f"{directory}/faiss.index")

    @classmethod
    def load(cls, directory, index_type="flat"):
        vecs = np.load(f"{directory}/embeddings.npy")
        st = cls(vecs.shape[1], index_type)
        import os
        if faiss is not None and os.path.exists(f"{directory}/faiss.index"):
            st.vecs, st.index = vecs, faiss.read_index(f"{directory}/faiss.index")
        else:
            st.add(vecs)
        return st
