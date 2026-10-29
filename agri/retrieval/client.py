import httpx
from agri.config import load_config

class RemoteRetriever:
    def __init__(self, url):
        self.url, self.http = url.rstrip("/"), httpx.Client(timeout=30)
        h = self.http.get(self.url + "/health").json(); self.kb_version = h["kb_version"]
        self.labels = self.http.get(self.url + "/labels").json()
    def search(self, query, k=None, crop="auto", mode=None, rerank=None, expand=True, parent_expand=None):
        body = {"query": query, "k": k or 3, "crop": crop, "mode": mode, "rerank": rerank, "expand": expand, "parent_expand": parent_expand}
        r = self.http.post(self.url + "/search", json=body); r.raise_for_status(); return r.json()
    def get_chunk(self, cid):
        r = self.http.get(f"{self.url}/chunk/{cid}"); return r.json() if r.status_code == 200 else None
    def expand_query(self, q): return q
    def detect_crop(self, q): return None

def get_retriever(cfg=None):
    cfg = cfg or load_config()
    if cfg["retrieval"]["mode"] == "remote":
        return RemoteRetriever(cfg["retrieval"]["remote_url"])
    from agri.retrieval.core import Retriever
    return Retriever(cfg=cfg)
