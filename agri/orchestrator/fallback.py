FALLBACK_TYPES = ("out_of_scope", "insufficient", "dose", "error", "unsupported_crop", "need_crop")

def make_fallback(kind, cfg, labels, chunks=None, question="", crop=None, unsupported=None):
    help_line = f"For reliable help, contact your local agriculture officer or Krishi Vigyan Kendra. {cfg['fallback']['helpline']}."
    crops = ", ".join(sorted({l["crop"] for l in labels}))
    if kind == "out_of_scope":
        msg = (f"I can only help with diseases and pests of these crops: {crops}. I can't answer this from my documents. {help_line}")
    elif kind == "unsupported_crop":
        name = f"'{unsupported}'" if unsupported else "This crop"
        msg = (f"{name} is not covered by my documents yet. I currently cover only: {crops}. Please do not rely on me for other crops. {help_line}")
    elif kind == "need_crop":
        msg = ("Which crop is this about? I cannot identify a problem without it. Please tell me the crop (" + crops + "), which plant part is affected "
               "(leaf, stem, fruit, grain), the colour and shape of the spots, and the recent weather. " + help_line)
    elif kind == "dose":
        msg = ("I can't give a spray dose because my documents do not contain a verified dose for this case. Doses depend on the product, "
               "crop stage and spray volume. Please follow the label of a registered product and ask your local agriculture officer. " + help_line)
    elif kind == "insufficient":
        topics = []
        for c in (chunks or []):
            if crop and c.get("crop") not in (crop, "General guidance"): continue   # never suggest topics of a crop the farmer did not name
            if c.get("disease") and f"{c['crop']} - {c['disease']}" not in topics: topics.append(f"{c['crop']} - {c['disease']}")
        near = f" The closest topics I found were: {', '.join(topics[:3])}." if topics else ""
        seen = f" I understood the crop as {crop}." if crop else ""
        ask = ("which plant part is affected (leaf, stem, fruit, grain), the colour and shape of the spots, how fast it is spreading and the recent weather"
               if crop else "the crop, which plant part is affected (leaf, stem, fruit, grain), the colour and shape of the spots, how fast it is spreading and the recent weather")
        msg = ("I could not find enough in my documents to identify this reliably." + seen + near + " Please tell me: " + ask + ". " + help_line)
    else:
        msg = "Something went wrong on our side. Please try again in a moment. " + help_line
    return {"type": kind, "message": msg}
