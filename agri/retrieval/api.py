"""Stateless retrieval microservice.  uvicorn agri.retrieval.api:app --port 8001"""
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agri.retrieval.core import Retriever

app = FastAPI(title="Retrieval service")

@lru_cache(maxsize=1)
def R():
    return Retriever()

class SearchReq(BaseModel):
    query: str = Field(min_length=2, max_length=20000)
    k: int = Field(3, ge=1, le=20)
    crop: str | None = "auto"
    mode: str | None = None
    rerank: bool | None = None
    expand: bool = True
    parent_expand: bool | None = None

@app.get("/health")
def health():
    r = R(); return {"status": "ok", "kb_version": r.kb_version, "n_chunks": len(r.chunks)}

@app.post("/search")
def search(req: SearchReq):
    return R().search(req.query, req.k, req.crop, req.mode, req.rerank, req.expand, req.parent_expand)

@app.get("/chunk/{chunk_id}")
def chunk(chunk_id: str):
    c = R().get_chunk(chunk_id)
    if not c: raise HTTPException(404, "unknown chunk")
    return c

@app.get("/labels")
def labels():
    return R().labels
