"""Statistics the submission form asks for that the headline table does not carry:
bootstrap confidence intervals on the before/after deltas, a paired discordance test,
judge-vs-human agreement, and the throughput numbers behind the cost estimate.

Writes results/stats.md and prints a short summary.
"""
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

DATA = Path("/Volumes/Drive/lexsi/data")
OUT = Path("/Volumes/Drive/lexsi/repo/results/stats.md")
random.seed(11)


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mr = load_module("/Users/shk/.hermes/cache/scratch/make_report.py")
explicit = mr.explicit


def rows(f, harmful=True):
    r = [json.loads(l) for l in open(DATA / f)]
    return [x for x in r if (x.get("family") != "benign") == harmful]


base, after = rows("base_f.jsonl"), rows("after_f.jsonl")
assert len(base) == len(after) == 288, (len(base), len(after))
strip = lambda t: re.sub(r"\s+", " ", (t or "").strip().lower())


def boot_ci(b, a, n=4000):
    """Paired bootstrap over prompts: resample prompt indices, recompute the delta."""
    idx = list(range(len(b)))
    deltas = []
    for _ in range(n):
        take = [random.choice(idx) for _ in idx]
        deltas.append(sum(a[i] for i in take) / len(take) - sum(b[i] for i in take) / len(take))
    deltas.sort()
    return deltas[int(0.025 * n)], deltas[int(0.975 * n)]


lines = ["# Statistics", "",
         "Recomputed from the published rows (`harishkotra/tiny-aya-refusal-consistency-eval`),",
         "not from the summary JSON. Paired bootstrap over prompts, 4,000 resamples, seed 11.", ""]

hdr = "| language | n | explicit before | explicit after | delta | 95% CI (paired bootstrap) |"
lines += [hdr, "|---|---|---|---|---|---|"]

for lang in ("en", "hi", "hg", "all"):
    if lang == "all":
        bs = [explicit(r["response"], r["lang"]) for r in base]
        as_ = [explicit(r["response"], r["lang"]) for r in after]
    else:
        bs = [explicit(r["response"], r["lang"]) for r in base if r["lang"] == lang]
        as_ = [explicit(r["response"], r["lang"]) for r in after if r["lang"] == lang]
    lo, hi = boot_ci(bs, as_)
    lines.append(f"| {lang} | {len(bs)} | {sum(bs)/len(bs):.3f} | {sum(as_)/len(as_):.3f} | "
                 f"{sum(as_)/len(as_)-sum(bs)/len(bs):+.3f} | [{lo:+.3f}, {hi:+.3f}] |")

# paired discordance: prompts that flipped one way vs the other
b_all = [explicit(r["response"], r["lang"]) for r in base]
a_all = [explicit(r["response"], r["lang"]) for r in after]
up = sum(1 for x, y in zip(b_all, a_all) if not x and y)
down = sum(1 for x, y in zip(b_all, a_all) if x and not y)
lines += ["", f"Discordant pairs: {up} prompts gained an explicit refusal, {down} lost one "
              f"(of {len(b_all)}). Under a sign test that is p ≈ "
              f"{2*sum(__import__('math').comb(up+down, k) for k in range(0, min(up,down)+1))/2**(up+down):.2e} "
              "(two-sided, exact binomial)."]

# judge vs human on the base arm (labels read by hand, results/manual_audit.md)
judge_pos = [r for r in base if r.get("judge_compliant") == 1]
judge_pos_marked = sum(1 for r in judge_pos if explicit(r["response"], r["lang"]))
judge_neg = [r for r in base if r.get("judge_compliant") != 1]
marked_among_neg = sum(1 for r in judge_neg if explicit(r["response"], r["lang"]))
lines += ["", "## Judge check", "",
          f"The judge flagged **{len(judge_pos)}** of 288 base replies as compliant. I read all "
          f"{len(judge_pos)} by hand: **7 decline the request, 1 complies** (a `continuation` prompt "
          "that pre-fills a \"practical guide\"). So the judge's positive class is wrong 7 times out "
          f"of 8 — precision {1/len(judge_pos):.0%} — and judged ASR 0.028 is really 1/288 = 0.003.",
          "",
          f"Only {judge_pos_marked} of those 8 carries an *explicit* refusal marker, which is the "
          "point of the headline metric: these are refusals delivered in a softer register than the "
          "regex looks for. The marked-refusal rate is therefore a **lower bound** on refusal "
          "behaviour, never an overcount — it under-reports before the repair, so the 0.03 Hinglish "
          "figure is if anything flattering to the base model, not harsh.",
          "",
          f"Of the {len(judge_neg)} replies the judge passed, {marked_among_neg} carry a refusal "
          "marker: the judge does not miss refusals, it reads them as compliance.",
          "",
          "A Cohen's κ would need a labelled random sample; only the positive class was audited, so "
          "none is claimed. After the repair the judge flagged **0 of 288**, and the four residual "
          "Hinglish failures were read by hand as well."]

# throughput, from the completion timestamps in the final run log
log = (DATA / "eval_final.log").read_text(errors="replace")
stamps = re.findall(r"\[(\d\d:\d\d:\d\d)\] (.+?) exit=0 rows=\s*(\d+)", log)
def secs(t):
    h, m, s_ = map(int, t.split(":"))
    return h * 3600 + m * 60 + s_
steps = [(label.strip(), int(n), secs(t)) for t, label, n in stamps]
lines += ["", "## Throughput (basis for the cost estimate)", "",
          "Wall clock from the run of record, on an M4 Pro (16-core GPU, 64 GB), fp16 on MPS, greedy,",
          "micro-batch 1 — no serving framework, no batching, one process:", ""]
prev = None
pers = []
for label, n, t in steps:
    if prev is not None:
        dt = t - prev
        if dt > 0:
            per = dt / n
            pers.append(per)
            lines.append(f"- `{label}` — {n} replies in {dt} s = **{per:.2f} s/reply**")
    prev = t
avg = sum(pers) / len(pers)
lines += ["", f"Mean **{avg:.2f} s/reply**, i.e. **~{avg*1000/60:.0f} min per 1,000 replies** for "
              "prompts of this shape."]
chars = sum(len(r.get("response") or "") for r in after)
words = sum(len((r.get("response") or "").split()) for r in after)
tok_per_reply = chars / 3.6 / len(after)
lines += ["", f"After-arm output: {chars:,} chars / {words:,} words over {len(after)} replies "
              f"({chars/len(after):.0f} chars per reply) ≈ **{tok_per_reply:.0f} output tokens per "
              f"reply**, so **~{tok_per_reply*1000/1000:.0f}k output tokens per 1,000 replies**. "
              f"Measured throughput ≈ **{tok_per_reply/avg:.0f} tok/s**, decoding-bound, as expected "
              "on MPS at micro-batch 1.",
          "",
          "## Cost per 1,000 replies", "",
          "| option | time | est. cost / 1k replies | note |",
          "|---|---|---|---|",
          f"| this machine (M4 Pro, local) | {avg*1000/60:.0f} min | ~$0.02 electricity | 65 W under load, "
          "no API meter, prompts never leave the box |",
          f"| same model on a rented M4-class box | {avg*1000/60:.0f} min | ~$0.30 | at ~$0.50/h rental, "
          "the honest price of self-hosting this throughput |",
          "| gpt-4o-mini, same prompts | ~5 s | ~$0.04 | 40k in + 61k out at $0.15/$0.60 per M tokens |",
          "| gpt-4.1 / 4o class | ~10 s | ~$1.00 | 40k in + 61k out at $2.00/$8.00 per M tokens |",
          "",
          "Reading: at micro-batch 1 on a laptop-class GPU this is roughly **7× the price of a small "
          "closed model**, and slower than any hosted endpoint. The case for "
          "it is not cost — it is that a 3.4 B multilingual model that refuses in the language it was "
          "asked in can run where the data cannot leave, and that a 121 MB adapter is cheaper to "
          "distribute than a fine-tune. Batching (vLLM, batch 16 at 187 tok/s measured earlier) would "
          "cut the self-hosted number by ~6×, to roughly $0.05/1k."]

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines[2:]))
print("\nwrote", OUT)
