"""Deterministic, document-grounded dataset builder (no LLM needed => reproducible & verifiable).
 - Every sample's label/crop/action come from the KB disease cards, never from a generator model.
 - Symptom statements are split per class: the last 2 are TEST-ONLY (reduces train/test wording leakage).
 - Deliberate imbalance: head classes get N_HEAD samples, rare classes N_RARE.
 - Special suites: long (512/1024/2048), terminology, OOD (eval and disjoint OOD-train).
python -m agri.datagen.build_dataset [--seed 7]"""
import argparse, json, random, re
from collections import defaultdict
from agri.config import load_config, P, write_jsonl, read_jsonl

HEAD = ["Early blight", "Late blight", "Rice blast", "Bacterial leaf blight", "Yellow rust", "Leaf rust", "Pink bollworm", "Tikka leaf spot"]
N_HEAD, N_RARE, HELD_OUT = 60, 8, 2
TEST_FRAC_HEAD, TEST_N_RARE = 15, 3     # head: 45 train / 15 test ; rare: 5 train / 3 test

TEMPL = ["my {crop} crop: {s}", "{crop} plants showing {s}", "in my {crop} field i see {s}. what is this disease?", "please help, {crop} - {s}",
         "{s} on my {crop}, what should i do", "farmer here. {crop} problem: {s}", "{crop} ka problem, {s}", "what disease is this? {s} ({crop})",
         "{s}. crop is {crop}", "my {crop} looks bad: {s}", "sir {crop} me {s}", "{crop} - {s}. how to treat"]
CONN = [", ", " and ", ". also ", " and also ", "; "]

FILLER = ["The market price of vegetables has been changing every week.", "I bought a new water pump last month from the town dealer.",
 "My son is studying in the city and calls every Sunday.", "The village road near the school was repaired recently.",
 "We are planning to attend the farmers meeting next month.", "I keep a notebook where I write all expenses of the farm.",
 "The tractor needed a new tyre and the mechanic took two days.", "There was a discussion about the new loan scheme at the cooperative bank.",
 "My neighbour has a small dairy and sells milk in the morning.", "The electricity supply comes only at night in our area.",
 "I took a bus to the district office to submit some papers.", "The local shop now accepts digital payments.",
 "We repaired the boundary wall of the farm house.", "The labourers come early in the morning and leave by evening.",
 "My wife prepares food for everyone working on the farm.", "I heard that the government will announce a new subsidy soon.",
 "We store our grain in a room near the house.", "The temple festival will be held in the village next week.",
 "I watched a programme on television about farm machinery.", "Our farm is about two kilometres away from the main road.",
 "I have been farming here for more than twenty years.", "The mobile network is very weak near the farm.",
 "My brother lives in another village and visits during festivals.", "The bank passbook needs to be updated this week.",
 "Some of the farmers here are thinking about buying a second hand harvester.", "The school children walk through our farm path every day.",
 "I need to renew my ration card soon.", "The cooperative society office opens at ten in the morning.",
 "A truck comes every Thursday to collect produce from the village.", "I met an agent who offered to buy the produce directly."]

TRADE = {"mancozeb": ["dithane", "indofil m-45"], "imidacloprid": ["confidor"], "copper oxychloride": ["blitox"], "propiconazole": ["tilt"],
         "tebuconazole": ["folicur"], "tricyclazole": ["beam"], "chlorothalonil": ["kavach"], "metalaxyl": ["ridomil"]}

OOD_EVAL = {
 "livestock": [f"my {a} {p}" for a, p in [("cow", "has high fever and is not eating"), ("buffalo", "is limping and the leg is swollen"), ("goat", "has loose motions and looks weak"),
              ("hen", "is not laying eggs and is dull"), ("sheep", "has a cough and runny nose"), ("cow", "gives very less milk since two days"), ("goat", "has swelling near the jaw"), ("buffalo", "is not coming in heat")]],
 "human": ["i have a headache and cough since three days", "my child has fever and rash, which medicine to give", "my stomach hurts after eating, what should i take",
           "how to control my blood sugar", "my father has chest pain what to do", "i have a tooth pain at night"],
 "unseen_crop": [f"my {c} {p}" for c, p in [("banana", "leaves have black streaks and are drying"), ("banana", "fruits are rotting on the plant"), ("mango", "flowers are turning black and falling"),
              ("mango", "fruits have dark spots and are rotting"), ("apple", "leaves have olive green scabs"), ("apple", "fruits are cracking with dark patches"),
              ("grape", "leaves have white powder on the upper side"), ("grape", "bunches are shrivelling and drying"), ("coffee", "leaves show orange powdery spots underneath"), ("coffee", "berries are dropping before ripening")]],
 "dose_unsupported": ["how much mancozeb should i mix in a 15 litre tank for tomato blight", "what is the exact dose of tricyclazole per acre for rice blast",
              "how many ml of imidacloprid per litre for tomato leaf curl", "tell me the dosage of propiconazole for wheat yellow rust per hectare",
              "how many grams of chlorothalonil in 10 litre water for groundnut tikka", "what quantity of copper oxychloride should i spray on cotton bacterial blight",
              "give me the mixing ratio of validamycin for sheath blight", "how much tebuconazole per tank for wheat leaf rust"],
 "general": ["what is the weather forecast for tomorrow", "who won the cricket match yesterday", "write a poem about rain", "what is the capital of australia",
             "how to invest in the stock market", "best phone under twenty thousand", "tell me a recipe for biryani", "how do i apply for a driving licence"]}
OOD_TRAIN = {
 "livestock": [f"my {a} {p}" for a, p in [("pig", "is not eating and shivering"), ("duck", "has watery eyes and is weak"), ("horse", "has a wound on the back leg"), ("camel", "is losing weight quickly"),
              ("rabbit", "has itchy skin and hair loss"), ("pig", "has red patches on the skin"), ("duck", "is not able to walk properly"), ("horse", "is coughing and breathing fast")]],
 "human": ["my mother has back pain since morning", "what to eat when i have a fever", "i feel dizzy and weak, which doctor should i see", "my baby is not sleeping at night",
           "how to reduce my blood pressure", "i have a skin rash on my arm"],
 "unseen_crop": [f"my {c} {p}" for c, p in [("onion", "bulbs are soft and smell bad"), ("onion", "leaf tips are turning white and dry"), ("papaya", "leaves are mottled and the plant is stunted"), ("papaya", "fruits have water soaked spots"),
              ("tea", "young shoots have grey blotches"), ("tea", "bushes have small blisters on the tender leaves"), ("maize", "cobs have black powdery masses"), ("maize", "long grey lesions on the leaves"),
              ("soybean", "pods are empty and the stems are discoloured"), ("soybean", "leaves have angular brown patches")]],
 "dose_unsupported": ["how many ml of mancozeb per litre for tomato early blight", "what dose of carbendazim should i give for rice brown spot per acre", "tell me the spray quantity of metalaxyl for tomato late blight",
              "how much tricyclazole to mix in 200 litre water", "what is the right dosage of imidacloprid for cotton whitefly per tank", "how many grams of hexaconazole per litre for sheath blight",
              "exact mixing quantity of tebuconazole for groundnut rust", "how much propiconazole in 15 litre sprayer for karnal bunt"],
 "general": ["what is the price of petrol today", "who is the prime minister of japan", "tell me a joke", "how to learn python", "translate hello to french",
             "which movie is best this week", "how to book a train ticket", "what is bitcoin"]}

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def misspell(text, rng):
    ws = text.split()
    idx = [i for i, w in enumerate(ws) if len(w) > 5 and w.isalpha()]
    for i in rng.sample(idx, min(len(idx), rng.choice([1, 2]))):
        w = ws[i]; j = rng.randrange(1, len(w) - 2)
        ws[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:] if rng.random() < 0.6 else w[:j] + w[j + 1:]
    return " ".join(ws)

def compose(stmts, crop_word, rng, noise=True):
    s = stmts[0]
    for x in stmts[1:]: s += rng.choice(CONN) + x
    t = rng.choice(TEMPL).format(crop=crop_word, s=s)
    if noise and rng.random() < 0.35: t = misspell(t, rng)
    if rng.random() < 0.25: t = re.sub(r"[.,;?()]", "", t)
    return t

def est_tokens(text, tok=None):
    return len(tok(text).input_ids) if tok else int(len(text.split()) * 1.35)

def get_tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        return None

def build(seed=7):
    cfg = load_config(); rng = random.Random(seed)
    out = P(cfg["paths"]["dataset_dir"]); (out / "special").mkdir(parents=True, exist_ok=True)
    kb = P(cfg["paths"]["kb_dir"]); cards = [c for c in json.loads((kb / "cards.json").read_text()) if c["type"] != "guidance"]
    gl = json.loads(P(cfg["paths"]["glossary"]).read_text()); tok = get_tok()
    train, test, seen, classes = [], [], set(), {}
    for c in cards:
        grp = "head" if c["disease"] in HEAD else "rare"
        stm = [s.strip() for s in c["fields"]["symptoms"]["text"].split(";") if s.strip()]
        tr_pool, te_only = stm[:-HELD_OUT], stm[-HELD_OUT:]
        action = (c["fields"]["cultural_control"]["text"] + " " + c["fields"]["chemical_control"]["text"]).strip()
        n_test = TEST_FRAC_HEAD if grp == "head" else TEST_N_RARE; n_train = (N_HEAD if grp == "head" else N_RARE) - n_test
        classes[c["disease"]] = {"crop": c["crop"], "group": grp, "n_train": n_train, "n_test": n_test, "card_id": c["card_id"]}
        cw = gl["crops"][c["crop"].lower()]
        for split, n in (("train", n_train), ("test", n_test)):
            made, tries = 0, 0
            while made < n and tries < 20000:
                tries += 1
                if split == "train": st = rng.sample(tr_pool, rng.choice([2, 3]))
                else:
                    st = [rng.choice(te_only)] + ([rng.choice(tr_pool)] if rng.random() < 0.5 else []); rng.shuffle(st)
                crop_word = c["crop"].lower() if rng.random() < 0.6 else rng.choice(cw)
                text = compose(st, crop_word, rng)
                if norm(text) in seen: continue
                seen.add(norm(text)); made += 1
                (train if split == "train" else test).append({"symptom_text": text, "crop": c["crop"], "diagnosis": c["disease"], "action": action, "group": grp, "split": split})
            assert made == n, f"could not build {n} unique {split} samples for {c['disease']}"
    rng.shuffle(train); rng.shuffle(test)
    for i, r in enumerate(train): r["id"] = f"tr{i:04d}"
    for i, r in enumerate(test): r["id"] = f"te{i:04d}"
    write_jsonl(out / "train.jsonl", train); write_jsonl(out / "test.jsonl", test)
    # handwritten (independent wording) - copied through for convenience
    hw = read_jsonl(P("data/handwritten_eval.jsonl"))
    for r in hw: r["group"] = classes[r["diagnosis"]]["group"]
    write_jsonl(out / "handwritten.jsonl", hw)
    # long suites: same 30 base items for all lengths (paired comparison)
    by = defaultdict(list)
    for r in test: by[r["diagnosis"]].append(r)
    base = [rng.choice(v) for v in by.values()]; rest = [r for r in test if r not in base]
    base += rng.sample(rest, 30 - len(base)); rng.shuffle(base)
    for L in (512, 1024, 2048):
        rows = []
        for r in base:
            sym = r["symptom_text"]; pos = rng.choice(["start", "middle", "end"]); need = L - est_tokens(sym, tok)
            fill, t = [], 0
            while t < need:
                s = rng.choice(FILLER); fill.append(s); t += est_tokens(s, tok)
            k = {"start": 0, "middle": len(fill) // 2, "end": len(fill)}[pos]
            text = " ".join(fill[:k] + [sym] + fill[k:])
            rows.append({**r, "id": r["id"] + f"_L{L}", "symptom_text": text, "target_tokens": L, "est_tokens": est_tokens(text, tok), "position": pos, "base_id": r["id"]})
        write_jsonl(out / "special" / f"long{L}.jsonl", rows)
    # terminology suite
    term = []
    for c in cards:
        stm = [s.strip() for s in c["fields"]["symptoms"]["text"].split(";") if s.strip()]; te_only = stm[-HELD_OUT:]
        chem = c["fields"]["chemical_control"]["text"].lower()
        trades = [(t, a) for a, ts in TRADE.items() if a in chem for t in ts]
        aliases = [a for a in c["aliases"] if a.lower() != c["disease"].lower()]
        for j in range(2 if c["disease"] not in HEAD[:4] else 3):
            s = rng.choice(te_only); cw = rng.choice(gl["crops"][c["crop"].lower()])
            if trades and j == 0:
                t, a = rng.choice(trades); text = f"{cw}: {s}. i already sprayed {t} but no change"; kind = "chemical_trade_name"; term_used = t
            else:
                al = rng.choice(aliases) if aliases else c["disease"]; text = f"{cw}: {s}. people here call it {al}, is that right?"; kind = "alias_or_local_name"; term_used = al
            term.append({"id": f"tm{len(term):03d}", "symptom_text": text, "crop": c["crop"], "diagnosis": c["disease"], "group": "head" if c["disease"] in HEAD else "rare", "term_kind": kind, "term": term_used})
    write_jsonl(out / "special" / "terminology.jsonl", term)
    # OOD suites
    def ood_rows(d, pre):
        rows = []
        for cat, items in d.items():
            for t in items: rows.append({"id": f"{pre}{len(rows):03d}", "symptom_text": t, "category": cat, "expected": "refuse"})
        return rows
    write_jsonl(out / "special" / "ood.jsonl", ood_rows(OOD_EVAL, "ood")); write_jsonl(out / "ood_train.jsonl", ood_rows(OOD_TRAIN, "oodtr"))
    (out / "classes.json").write_text(json.dumps(classes, indent=2))
    stats = {"seed": seed, "train": len(train), "test": len(test), "handwritten": len(hw), "terminology": len(term), "ood": sum(len(v) for v in OOD_EVAL.values()),
             "ood_train": sum(len(v) for v in OOD_TRAIN.values()), "n_classes": len(classes), "head_classes": sum(v["group"] == "head" for v in classes.values()), "tokenizer_used": bool(tok)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2)); print(json.dumps(stats))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7); build(ap.parse_args().seed)
