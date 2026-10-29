"""Streamlit front end. Deploy: Streamlit Community Cloud, main file app/streamlit_app.py, secrets = LLM_API_KEY, LLM_BASE_URL, LLM_MODEL."""
import html, json, os, sys, time
from pathlib import Path
from urllib.parse import quote
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import streamlit as st
for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
    try:
        if k in st.secrets: os.environ[k] = str(st.secrets[k])
    except Exception: pass
from agri.config import load_config, load_variants

st.set_page_config(page_title="Crop Disease Advisory", page_icon="🌾", layout="wide")

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_pipeline():
    from agri.orchestrator.graph import Pipeline
    return Pipeline()

def read_doc(path):
    p = ROOT / path
    return p.read_text(encoding="utf-8") if p.exists() and p.suffix in (".md", ".txt") else None

def highlighted(text, start, end):
    if start is None or end is None: return html.escape(text).replace("\n", "<br>")
    return (html.escape(text[:start]) + "<mark>" + html.escape(text[start:end]) + "</mark>" + html.escape(text[end:])).replace("\n", "<br>")

def source_card(s, key):
    with st.container(border=True):
        st.markdown(f"**[{s['id']}] {s['crop']} · {s['disease'] or s['title']} · {s['section']}**")
        st.caption(s["title"] + (f" (page {s['page']})" if s.get("page") else ""))
        st.markdown(f"> {s['text']}")
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.markdown(f"[Permalink](?chunk={quote(s['chunk_id'])})")
        if s.get("url"): c2.link_button("Original source", s["url"])
        if st.toggle("Show in full source document", key=f"t_{key}_{s['id']}"):
            doc = read_doc(s["doc_path"])
            if doc: st.markdown(f"<div style='font-size:0.85rem;line-height:1.5'>{highlighted(doc, s['start'], s['end'])}</div>", unsafe_allow_html=True)
            else: st.info(f"Document file: {s['doc_path']}")

def viewer(pipe, chunk_id):
    c = pipe.R.get_chunk(chunk_id)
    st.title("Source viewer")
    if not c: st.error("Unknown source id."); return
    source_card({"id": "S", "chunk_id": c["chunk_id"], "crop": c["crop"], "disease": c["disease"], "title": c["source_title"], "section": c["section_name"], "text": c["raw_text"],
                 "doc_path": c["doc_path"], "page": c["page"], "start": c["start"], "end": c["end"], "url": c["source_url"]}, "viewer")
    st.markdown("[← Back to assistant](./)")

def ratelimit():
    now = time.time(); q = [t for t in st.session_state.get("q", []) if now - t < 600]
    st.session_state["q"] = q
    if len(q) >= 15: return False
    q.append(now); return True

pipe = get_pipeline()
if st.query_params.get("chunk"):
    viewer(pipe, st.query_params["chunk"]); st.stop()

st.title("🌾 Crop Disease Diagnosis & Treatment Advisory")
st.caption("Grounded in a curated knowledge base. Every answer cites its sources; if the documents don't support an answer, the assistant says so instead of guessing.")
variants = load_variants(); allowed = ["llm_api", "ollama"] + (["fp16", "nf4", "qlora_nf4", "lora_fp16", "lora_merged_nf4", "int8"] if os.environ.get("AGRI_ENABLE_LOCAL") else []) + (["mock"] if os.environ.get("AGRI_ENABLE_MOCK") else [])
with st.sidebar:
    variant = st.selectbox("Model backend", [v for v in allowed if v in variants], help="The cloud demo uses the hosted LLM API; local quantized models are used for the evaluation study.")
    st.markdown(f"**KB version:** `{pipe.R.kb_version}`")
    crops = sorted({l["crop"] for l in pipe.labels}); st.markdown("**Crops covered:** " + ", ".join(crops))
    st.markdown("**Diseases/pests covered:** " + str(len(pipe.labels)))
    st.warning("Advisory only. Never follow a pesticide dose from any chatbot: use the product label and your local agriculture officer.")
EXAMPLES = ["My tomato leaves have brown rings like a target and yellow patches around them", "gehun ke patton par peele rang ki line me powder hai", "Cotton bolls have small holes and pink worms inside",
            "How much mancozeb should I mix in a 15 litre tank?", "My cow has fever and is not eating"]
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(ex[:38] + "...", key=f"ex{i}", use_container_width=True): st.session_state["text"] = ex
text = st.text_area("Describe the problem (crop, plant part, what you see, weather):", key="text", height=110, max_chars=2000)
if st.button("Get advice", type="primary"):
    t = (text or "").strip()
    if len(t) < 5: st.error("Please describe the problem in a few words (at least 5 characters).")
    elif not ratelimit(): st.error("Too many requests. Please wait a few minutes.")
    else:
        with st.spinner("Searching the knowledge base and drafting an answer..."):
            res = pipe.run(t, variant)
        st.session_state["res"] = res; st.session_state["rid"] = st.session_state.get("rid", 0) + 1
        (ROOT / "logs").mkdir(exist_ok=True)
        with open(ROOT / "logs/requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "variant": variant, "status": res["status"], "latency_ms": round(res["latency_ms"]), "attempts": res["attempts"], "kb": res["kb_version"]}) + "\n")
res = st.session_state.get("res")
if res:
    rid = st.session_state["rid"]
    if res["status"] == "answered":
        st.success(f"**Likely problem: {res['diagnosis']}** ({res['crop']})")
        st.markdown(res["advice"])
        if res["needs_expert"]: st.warning("The assistant is not fully certain. Please confirm with your local agriculture officer.")
        st.subheader("Sources")
        for s in res["sources"]: source_card(s, rid)
    else:
        st.warning(res["message"])
    with st.expander("How this answer was produced (trace)"):
        st.write({"kb_version": res["kb_version"], "backend": res["variant"], "attempts": res["attempts"], "latency_ms": round(res["latency_ms"]), "review_reasons": res["reasons"]})
        st.json(res["trace"])
    c1, c2, _ = st.columns([1, 1, 6])
    for lab, val in (("👍", 1), ("👎", -1)):
        if (c1 if val == 1 else c2).button(lab, key=f"fb{val}_{rid}"):
            with open(ROOT / "logs/feedback.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps({"t": time.time(), "rid": rid, "value": val, "status": res["status"]}) + "\n")
            st.toast("Thanks for the feedback")
