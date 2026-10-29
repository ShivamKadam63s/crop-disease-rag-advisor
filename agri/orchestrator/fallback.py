FALLBACK_TYPES = ("out_of_scope", "insufficient", "dose", "error")

def make_fallback(kind, cfg, labels, chunks=None, question=""):
    help_line = f"For reliable help, contact your local agriculture officer or Krishi Vigyan Kendra. {cfg['fallback']['helpline']}."
    crops = ", ".join(sorted({l["crop"] for l in labels}))
    if kind == "out_of_scope":
        msg = (f"I can only help with diseases and pests of these crops: {crops}. I can't answer this from my documents. {help_line}")
    elif kind == "dose":
        msg = ("I can't give a spray dose because my documents do not contain a verified dose for this case. Doses depend on the product, "
               "crop stage and spray volume. Please follow the label of a registered product and ask your local agriculture officer. " + help_line)
    elif kind == "insufficient":
        topics = []
        for c in (chunks or []):
            if c.get("disease") and f"{c['crop']} - {c['disease']}" not in topics: topics.append(f"{c['crop']} - {c['disease']}")
        near = f" The closest topics I found were: {', '.join(topics[:3])}." if topics else ""
        msg = ("I could not find enough in my documents to identify this reliably." + near +
               " Please tell me: the crop, which plant part is affected (leaf, stem, fruit, grain), the colour and shape of the spots, "
               "how fast it is spreading and the recent weather. " + help_line)
    else:
        msg = "Something went wrong on our side. Please try again in a moment. " + help_line
    return {"type": kind, "message": msg}
