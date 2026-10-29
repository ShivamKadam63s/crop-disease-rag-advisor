"""Generation microservice.  uvicorn agri.generation.api:app --port 8002"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agri.generation.backends import get_backend, GenerationError

app = FastAPI(title="Generation service")

class GenReq(BaseModel):
    variant: str
    messages: list[dict] = Field(min_length=1)
    max_new_tokens: int = Field(220, ge=1, le=1024)

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/generate")
def generate(req: GenReq):
    try:
        return get_backend(req.variant).generate(req.messages, req.max_new_tokens)
    except GenerationError as e:
        raise HTTPException(502, str(e))
    except Exception:
        raise HTTPException(502, "generation failed")
