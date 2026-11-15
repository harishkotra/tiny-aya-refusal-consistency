"""Fill the article placeholders from the metrics file (default: metrics_final.json).

The article never hand-copies a number: every table below is regenerated from the same
metrics file the report and the dataset card use.
"""
import json, os, re

D = "/Volumes/Drive/lexsi/data"
MET = os.environ.get("METRICS", f"{D}/metrics_final.json")
M = json.load(open(MET))
ART = "/Volumes/Drive/lexsi/repo/article.md"

ORDER = [("asr", "attack success (judge, lower better)"),
         ("explicit", "explicit refusal in the asked language (higher better)"),
         ("degenerate", "repetition collapse (lower better)"),
         ("trunc", "no stop token / runaway (lower better)"),
         ("unique", "distinct replies (higher better)"),
         ("len_ch", "mean reply length, chars")]


def table(block):
    b = block["before"]["overall"] if "overall" in block["before"] else block["before"]
    a = block["after"]["overall"] if "overall" in block["after"] else block["after"]
    L = ["| metric | before | after | delta |", "|---|---|---|---|"]
    for k, label in ORDER:
        bv, av = b.get(k), a.get(k)
        if isinstance(bv, int) and isinstance(av, int):
            d = f"{av - bv:+d}"
        else:
            d = f"{av - bv:+.3f}"
        L.append(f"| {label} | {bv} | {av} | {d} |")
    return "\n".join(L)


def by_lang():
    L = ["| cell | explicit refusal before | after | collapse before | after | distinct replies after |",
         "|---|---|---|---|---|---|"]
    names = {"en": "English", "hi": "Hindi (Devanagari)", "hg": "Hinglish (romanised)"}
    for k in ("en", "hi", "hg"):
        b = M["harmful"]["before"]["by_lang"][k]
        a = M["harmful"]["after"]["by_lang"][k]
        L.append(f"| {names[k]} | {b['explicit']:.2f} | {a['explicit']:.2f} | "
                 f"{b['degenerate']:.2f} | {a['degenerate']:.2f} | {a.get('unique', 1):.2f} |")
    return "\n".join(L)


def benign():
    b = M["benign"]["before"]["by_lang"]
    a = M["benign"]["after"]["by_lang"]
    L = ["| cell | benign prompts answered with a refusal-shaped reply |", "|---|---|---|", "| | before | after |", "|---|---|---|"]
    for k in ("en", "hi", "hg"):
        L.append(f"| {k} | {b[k]['explicit']:.3f} | {a[k]['explicit']:.3f} |")
    return "\n".join(L)


def rounds():
    """Every repair round that was actually evaluated on the Hinglish cell."""
    import glob
    cast = [("v1 template", "metrics.json"),
            ("v2 long-form", "metrics_v2.json"),
            ("v3 gated", "metrics_v3.json"),
            ("v4 short + rep-pen 1.10", "metrics_final.json")]
    L = ["| round | hg explicit | hg collapse | hg distinct | training rows |",
         "|---|---|---|---|---|"]
    rows = {"v1 template": 404, "v2 long-form": 405, "v3 gated": 392, "v4 short + rep-pen 1.10": 380}
    for name, f in cast:
        p = f"{D}/{f}"
        if not os.path.exists(p):
            continue
        m = json.load(open(p))["harmful"]["after"]["by_lang"]["hg"]
        L.append(f"| {name} | {m['explicit']:.2f} | {m['degenerate']:.2f} | {m.get('unique', 0):.2f} | {rows[name]} |")
    return "\n".join(L)


body = open(ART).read()
body = re.sub(r"<!-- TABLE:headline -->.*?<!-- /TABLE -->", "<!-- TABLE:headline -->\n"
              + table(M["harmful"]) + "\n<!-- /TABLE -->", body, flags=re.S)
body = re.sub(r"<!-- TABLE:lang -->.*?<!-- /TABLE -->", "<!-- TABLE:lang -->\n"
              + by_lang() + "\n<!-- /TABLE -->", body, flags=re.S)
body = re.sub(r"<!-- TABLE:benign -->.*?<!-- /TABLE -->", "<!-- TABLE:benign -->\n"
              + benign() + "\n<!-- /TABLE -->", body, flags=re.S)
body = re.sub(r"<!-- TABLE:rounds -->.*?<!-- /TABLE -->", "<!-- TABLE:rounds -->\n"
              + rounds() + "\n<!-- /TABLE -->", body, flags=re.S)
open(ART, "w").write(body)
print(f"filled {ART} from {MET} | harmful after explicit={M['harmful']['after']['overall']['explicit']}")
