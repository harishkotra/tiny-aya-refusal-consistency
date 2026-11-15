"""Fill the README results block from the metrics file. Tables are never hand-copied."""
import json, os, re
from pathlib import Path

D = Path("/Volumes/Drive/lexsi/data")
M = json.load(open(os.environ.get("METRICS", D / "metrics_final.json")))
README = Path("/Volumes/Drive/lexsi/repo/README.md")
NAMES = {"en": "English", "hi": "Devanagari Hindi", "hg": "Hinglish"}


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "-"


L = ["| cell | explicit refusal | | collapse | | distinct replies | | judged ASR |",
     "|---|---|---|---|---|---|---|---|",
     "| | before | after | before | after | before | after | after |",
     "|---|---|---|---|---|---|---|---|"]
for k in ("en", "hi", "hg"):
    b = M["harmful"]["before"]["by_lang"][k]
    a = M["harmful"]["after"]["by_lang"][k]
    L.append(f"| {NAMES[k]} | {fmt(b['explicit'])} | {fmt(a['explicit'])} | {fmt(b['degenerate'])} | "
             f"{fmt(a['degenerate'])} | {fmt(b.get('unique'))} | {fmt(a.get('unique'))} | {fmt(a['asr'])} |")

L += ["", "Benign control (share of benign prompts answered with a refusal-shaped reply — the "
      "over-refusal guard):", "",
      "| cell | before | after |", "|---|---|---|"]
for k in ("en", "hi", "hg"):
    b = M["benign"]["before"]["by_lang"][k]
    a = M["benign"]["after"]["by_lang"][k]
    L.append(f"| {NAMES[k]} | {fmt(b['explicit'])} | {fmt(a['explicit'])} |")

body = README.read_text()
body = re.sub(r"<!-- RESULTS_TABLE -->.*?(?=\n## )", "<!-- RESULTS_TABLE -->\n\n" + "\n".join(L) + "\n\n",
              body, flags=re.S)
README.write_text(body)
print(f"filled {README} | headline after explicit={M['harmful']['after']['overall']['explicit']}")
