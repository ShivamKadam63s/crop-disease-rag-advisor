import httpx
from agri.generation.backends import Backend, GenerationError

class RemoteBackend(Backend):
    def __init__(self, url, variant):
        self.url, self.variant, self.name = url.rstrip("/"), variant, variant
    def generate(self, messages, max_new_tokens=220):
        try:
            r = httpx.post(self.url + "/generate", json={"variant": self.variant, "messages": messages, "max_new_tokens": max_new_tokens}, timeout=300)
        except httpx.HTTPError:
            raise GenerationError("generation service unreachable")
        if r.status_code != 200: raise GenerationError("generation service error")
        return r.json()

def resolve_backend(variant, cfg):
    if cfg["generation"]["mode"] == "remote": return RemoteBackend(cfg["generation"]["remote_url"], variant)
    from agri.generation.backends import get_backend
    return get_backend(variant)
