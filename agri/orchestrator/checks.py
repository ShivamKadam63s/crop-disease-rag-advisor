"""Review step: decides whether a drafted answer may be shown."""
from agri.generation.parse import REFUSALS
from agri.textutils import doses_in, support_score

def review(parsed, ctx, labels, dense_top, dcfg):
    """Returns dict(ok, reasons, refusal, support, dose_violations, citation_ok)."""
    reasons, r = [], {"refusal": None}
    ctx_by_id = {f"S{i}": c for i, c in enumerate(ctx, 1)}
    if not parsed["parse_ok"]: reasons.append("invalid_json")
    lab = parsed["label"]
    if lab in REFUSALS:
        r["refusal"] = lab; reasons.append(lab.lower())
    elif lab is None: reasons.append("unknown_label")
    else:
        crops = {l["crop"] for l in labels if l["disease"] == lab}
        if parsed["crop"] and parsed["crop"] not in crops: reasons.append("crop_label_mismatch")
    cites = parsed["citations"]
    cited = [ctx_by_id[c] for c in cites if c in ctx_by_id]
    if lab and lab not in REFUSALS:
        if not cites: reasons.append("no_citation")
        elif len(cited) != len(cites): reasons.append("invalid_citation_id")
        elif not any(c["disease"] == lab for c in cited): reasons.append("citation_disease_mismatch")
    texts = [c["raw_text"] for c in cited] or [c["raw_text"] for c in ctx]
    sup = support_score(parsed["advice"], texts) if parsed["advice"] else 0.0
    if lab and lab not in REFUSALS and sup < dcfg["min_support"]: reasons.append("low_support")
    ctx_doses = set().union(*[doses_in(c["raw_text"]) for c in ctx]) if ctx else set()
    viol = sorted(doses_in(parsed["advice"]) - ctx_doses)
    if viol: reasons.append("unsupported_dose")
    if dense_top < dcfg["min_dense_score"]: reasons.append("weak_retrieval")
    r.update({"ok": not reasons, "reasons": reasons, "support": sup, "dose_violations": viol,
              "citation_ok": ("no_citation" not in reasons and "invalid_citation_id" not in reasons and "citation_disease_mismatch" not in reasons)})
    return r
