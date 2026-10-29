"""Build the knowledge base: ingest -> parse -> chunk -> validate -> embed -> index -> manifest.
   python -m agri.kb.build [--strategy structured|naive] [--max_words N] [--overlap N] [--embedder MODEL] [--backend B] [--index flat|hnsw] [--out DIR]"""
import argparse, hashlib, json, time
from pathlib import Path
from agri.config import load_config, P, read_jsonl, write_jsonl
from agri.kb import ingest, chunking, validate as V
from agri.retrieval.embedder import from_cfg
from agri.retrieval.store import VectorStore

def build(cfg, strategy=None, max_words=None, overlap=None, model=None, backend=None, index_type=None, out=None, quiet=False):
    k = cfg["kb"]
    strategy = strategy or k["strategy"]; max_words = max_words or k["chunk_max_words"]; overlap = k["chunk_overlap_words"] if overlap is None else overlap
    out = Path(out or P(cfg["paths"]["kb_dir"])); out.mkdir(parents=True, exist_ok=True)
    glossary = json.loads(P(cfg["paths"]["glossary"]).read_text())
    docs = ingest.load_documents(P(cfg["paths"]["raw_docs"]))
    cards, chunks = [], []
    for d in docs:
        if ingest.is_structured(d):
            cs = ingest.parse_structured(d); cards += cs
            for c in cs:
                chunks += (chunking.chunk_card_structured(c, max_words, overlap, k["header_in_chunk"]) if strategy == "structured"
                           else chunking.chunk_card_naive(c, max_words, overlap))
        else:
            chunks += chunking.chunk_generic(d, ingest.infer_crop(d, glossary), max_words, overlap)
    report = V.validate(cards, chunks, docs, glossary, max_words)
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit("KB validation FAILED:\n" + "\n".join(report["errors"][:30]))
    emb = from_cfg(cfg["retrieval"], model, backend)
    vecs = emb.encode_passages([c["text"] for c in chunks])
    store = VectorStore(vecs.shape[1], index_type or cfg["retrieval"]["index_type"]); store.add(vecs); store.save(str(out))
    write_jsonl(out / "chunks.jsonl", chunks)
    labels = [{"crop": c["crop"], "disease": c["disease"], "card_id": c["card_id"], "aliases": c["aliases"]} for c in cards if c["type"] != "guidance"]
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    (out / "cards.json").write_text(json.dumps(cards, indent=2))
    manifest = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "strategy": strategy, "chunk_max_words": max_words, "overlap": overlap,
                "embedder_model": emb.model_name, "embedder_backend": emb.backend, "dim": emb.dim, "index_type": store.index_type,
                "n_chunks": len(chunks), "n_cards": len(cards),
                "docs": {d["doc_id"]: d["sha256"] for d in docs}}
    manifest["kb_version"] = hashlib.sha256(json.dumps(manifest["docs"], sort_keys=True).encode() + f"{strategy}{max_words}{overlap}{emb.model_name}".encode()).hexdigest()[:12]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not quiet:
        print(f"KB built: {len(chunks)} chunks, {len(labels)} labels, version {manifest['kb_version']}, warnings: {len(report['warnings'])} -> {out}")
    return manifest

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy"); ap.add_argument("--max_words", type=int); ap.add_argument("--overlap", type=int)
    ap.add_argument("--embedder"); ap.add_argument("--backend"); ap.add_argument("--index"); ap.add_argument("--out")
    a = ap.parse_args()
    build(load_config(), a.strategy, a.max_words, a.overlap, a.embedder, a.backend, a.index, a.out)
