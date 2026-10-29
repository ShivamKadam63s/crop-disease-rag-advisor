"""Dataset verification: python -m agri.datagen.verify   (exit code 1 on any ERROR)
Writes data/dataset/verification_report.json"""
import json, re, sys
from collections import Counter
from agri.config import load_config, P, read_jsonl
from agri.datagen.build_dataset import FILLER, N_HEAD, N_RARE, TEST_FRAC_HEAD, TEST_N_RARE, OOD_EVAL, OOD_TRAIN, norm
from agri.textutils import jaccard

FORBID = ["spot", "rust", "blight", "wilt", "curl", "mildew", "smut", "bunt", "rot ", "whitefly", "caterpillar", "leaves", "leaf", "yellow", "brown", "black", "powder", "blast", "bollworm", "worm"]

def main():
    cfg = load_config(); d = P(cfg["paths"]["dataset_dir"]); kb = P(cfg["paths"]["kb_dir"])
    labels = {l["disease"]: l["crop"] for l in json.loads((kb / "labels.json").read_text())}
    gl = json.loads(P(cfg["paths"]["glossary"]).read_text())
    cards = {c["disease"]: c for c in json.loads((kb / "cards.json").read_text())}
    classes = json.loads((d / "classes.json").read_text())
    err, warn, info = [], [], {}
    sets = {n: read_jsonl(d / f"{n}.jsonl") for n in ("train", "test", "handwritten", "ood_train")}
    for n in ("long512", "long1024", "long2048", "terminology", "ood"): sets[n] = read_jsonl(d / "special" / f"{n}.jsonl")
    # schema + label/crop consistency
    for n, rows in sets.items():
        if n.startswith("ood"): 
            for r in rows:
                if not r.get("symptom_text") and not r.get("text"): err.append(f"{n}:{r.get('id')} empty text")
            continue
        for r in rows:
            if r["diagnosis"] not in labels: err.append(f"{n}:{r['id']} unknown label {r['diagnosis']}")
            elif labels[r["diagnosis"]] != r["crop"]: err.append(f"{n}:{r['id']} crop/label mismatch")
            if len(r["symptom_text"].split()) < 4: err.append(f"{n}:{r['id']} too short")
    # action is traceable to KB card text
    for r in sets["train"] + sets["test"]:
        c = cards[r["diagnosis"]]
        if r["action"] != (c["fields"]["cultural_control"]["text"] + " " + c["fields"]["chemical_control"]["text"]).strip(): err.append(f"{r['id']} action not equal to KB source")
    # duplicates + leakage
    for n in ("train", "test", "handwritten"):
        c = Counter(norm(r["symptom_text"]) for r in sets[n])
        for k, v in c.items():
            if v > 1: err.append(f"{n}: duplicate text '{k[:50]}'")
    tr = {norm(r["symptom_text"]) for r in sets["train"]}
    for n in ("test", "handwritten"):
        for r in sets[n]:
            if norm(r["symptom_text"]) in tr: err.append(f"LEAK {n}:{r['id']} identical to a train sample")
    mx = 0.0
    for r in sets["test"][:150]:
        mx = max(mx, max(jaccard(r["symptom_text"], t["symptom_text"]) for t in sets["train"] if t["diagnosis"] == r["diagnosis"]))
    info["max_test_train_jaccard_same_class"] = round(mx, 3)
    if mx > 0.9: warn.append("some test samples are near-duplicates of train samples")
    # class counts / imbalance design
    for cls, meta in classes.items():
        ntr = sum(r["diagnosis"] == cls for r in sets["train"]); nte = sum(r["diagnosis"] == cls for r in sets["test"])
        exp_tr = (N_HEAD - TEST_FRAC_HEAD) if meta["group"] == "head" else (N_RARE - TEST_N_RARE); exp_te = TEST_FRAC_HEAD if meta["group"] == "head" else TEST_N_RARE
        if (ntr, nte) != (exp_tr, exp_te): err.append(f"{cls}: counts train/test {ntr}/{nte} != {exp_tr}/{exp_te}")
    if set(classes) != set(labels): err.append("classes.json and KB labels differ")
    # special sets
    for L in (512, 1024, 2048):
        rows = sets[f"long{L}"]
        if len(rows) != 30: err.append(f"long{L}: expected 30 rows")
        for r in rows:
            if abs(r["est_tokens"] - L) > 0.12 * L: err.append(f"long{L}:{r['id']} est_tokens {r['est_tokens']} off target")
    for s in FILLER:
        for f in FORBID:
            if f in s.lower() + " ": err.append(f"filler contains diagnostic term '{f}': {s}")
    tm_terms = set(gl["chemical_aliases"]) | {a.lower() for c in cards.values() for a in c["aliases"]} | {c["disease"].lower() for c in cards.values()}
    for r in sets["terminology"]:
        if not any(t in r["symptom_text"].lower() for t in tm_terms): err.append(f"terminology:{r['id']} contains no glossary/alias term")
    if len(sets["terminology"]) < 36: err.append("terminology set too small")
    # OOD disjointness and no accidental in-scope content
    ev = {norm(r["symptom_text"]) for r in sets["ood"]}; trn = {norm(r["symptom_text"]) for r in sets["ood_train"]}
    if ev & trn: err.append("OOD eval/train overlap")
    if len(sets["ood"]) < 40: err.append("OOD eval < 40 items")
    info["counts"] = {k: len(v) for k, v in sets.items()}
    info["ood_categories"] = dict(Counter(r["category"] for r in sets["ood"]))
    info["rare_test_samples_per_class"] = TEST_N_RARE
    warn.append(f"Rare classes have only {TEST_N_RARE} test samples each: per-class F1 is noisy; report with this caveat.")
    warn.append("Train/test texts are template-composed from KB symptom statements (last 2 per class are test-only). Use the handwritten set as the independent check; expect optimistic numbers on 'test'.")
    rep = {"errors": err, "warnings": warn, "info": info}
    (d / "verification_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(info, indent=1)); [print("WARN:", w) for w in warn]
    if err:
        print(f"\n{len(err)} ERRORS"); [print(" -", e) for e in err[:40]]; sys.exit(1)
    print("\nDATASET VERIFIED: 0 errors")

if __name__ == "__main__":
    main()
