"""Photo -> short symptom description (NOT a diagnosis). The text is shown to the farmer, who can edit it before it goes through the
normal retrieval + guard + review pipeline, so the diagnosis is always grounded in the knowledge base and never in the image model.
Needs an OpenAI-compatible vision model: LLM_VISION_MODEL (default: tries qwen/qwen3.8-27b, then meta-llama/llama-4-scout-17b-16e-instruct, on Groq)."""
import base64, os
from agri.generation.backends import GenerationError

PROMPT = ("Look at this photo of a plant. In ONE or TWO plain sentences describe only what is visibly wrong: the plant part (leaf, stem, fruit, boll, grain), "
          "colours, shapes and pattern of spots or damage, and any insects visible. If you can tell the crop, name it. Do NOT name a disease and do NOT give advice. "
          "If the photo does not show a plant, reply exactly: NOT_A_PLANT.")

def describe_photo(data: bytes, mime="image/jpeg"):
    import requests
    key = os.environ.get("LLM_API_KEY", ""); base = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    models = [m for m in (os.environ.get("LLM_VISION_MODEL"), "qwen/qwen3.8-27b", "meta-llama/llama-4-scout-17b-16e-instruct") if m]
    if not key or "your_" in key: raise GenerationError("LLM_API_KEY is not set.")
    if len(data) > 4_000_000: raise GenerationError("Photo is too large (max 4 MB).")
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    r = None
    for model in models:   # first model this key can use wins (404/400 = not available or not a vision model)
        body = {"model": model, "temperature": 0, "max_tokens": 200,
                "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}]}
        try:
            r = requests.post(base + "/chat/completions", json=body, timeout=60, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        except requests.RequestException:
            raise GenerationError("Photo service not reachable.")
        if r.status_code not in (400, 404): break
    if r is None or r.status_code != 200: raise GenerationError(f"Photo service returned status {getattr(r, 'status_code', 'n/a')}")
    text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    if not text or text.upper().startswith("NOT_A_PLANT"): raise GenerationError("I could not see a plant in this photo. Please take a clear close-up of the affected leaf or fruit.")
    return text
