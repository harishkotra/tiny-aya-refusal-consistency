"""Curate the generated repair set through CuratorKIT.

Stage 1 of the chain: CuratorKIT gates -> SafeTune-style LoRA repair -> AuditKIT scoring.

The gate is the interesting part. Generated refusals fail in three ways that look fine
to a human skim: they keep the harmful noun and then explain it anyway, they leak a
numbered procedure, or they are so short they are not a refusal so much as a shrug.
For benign samples the failure is the opposite - a stray refusal would teach over-refusal.
"""
import json, os, re, sys
from pathlib import Path

from curatorkit.pipeline import Pipeline, BaseGate, BaseExporter, BaseReader, BaseNormalizer
from curatorkit.exporters import MessagesExporter
from curatorkit.schema import DataSample, RejectedSample

SRC = "/Volumes/Drive/lexsi/data/repair_sft.jsonl"
FIXES = "/Volumes/Drive/lexsi/data/repair_sft_fixed.jsonl"
HG_DIVERSE = "/Volumes/Drive/lexsi/data/repair_hg_diverse.jsonl"  # takes precedence for hg
OUTDIR = os.environ.get("OUTDIR", "/Volumes/Drive/lexsi/data/curated")
REJECT_LOG = os.environ.get("REJECT_LOG", "/Volumes/Drive/lexsi/data/curation_diagnostics.json")
CURATED = os.environ.get("CURATED", "/Volumes/Drive/lexsi/data/repair_sft_curated.jsonl")

MARKERS = {
    "en": r"\b(can't|cannot|can not|won't|will not|unable to|not able to|i'm not going to|refuse|not going to)\b",
    "hi": r"(नहीं (दे|बता|कर|सका|करूंगा|कर सकता|कर सकती|मदद)|माफ|क्षमा|अस्वीकार|इनकार|नहीं कर सकता|नहीं बता सकता)",
    # Hinglish refusals vary a lot in phrasing, so match "nahi ... sakta"-shaped clauses
    # rather than a fixed list of sentences.
    "hg": r"\bnahi\b[^.]{0,60}\b(sakt[aie]|paunga|sakunga)\b|\b(maaf kijiye|maafi|mana kar)\b",
}
HOWTO = ("step 1", "ingredients:", "materials:", "you will need", "here's how", "here is how",
         "first,", "pehle,", "चरण 1", "सामग्री")


def mk_rej(sample, reason):
    return RejectedSample(id=sample.id, source_uri=sample.source_uri,
                          rejection_reason=reason, rejecting_step="RefusalQualityGate")


class RepairReader(BaseReader):
    def __init__(self, path):
        self.path = path

    def read(self):
        fixes = {}
        if os.path.exists(FIXES):
            for line in open(FIXES):
                r = json.loads(line)
                fixes[r["key"]] = r["text"]
        if os.path.exists(HG_DIVERSE):
            for line in open(HG_DIVERSE):
                r = json.loads(line)
                fixes[r["key"]] = r["text"]
        out, rej = [], []
        for i, line in enumerate(open(self.path)):
            try:
                r = json.loads(line)
            except Exception as e:
                rej.append(RejectedSample(id=f"line{i}", source_uri=self.path, rejection_reason=f"bad json: {e}", rejecting_step="RepairReader"))
                continue
            u, a = r["messages"][0]["content"], fixes.get(r["key"], r["messages"][1]["content"])
            out.append(DataSample(id=r["key"], source_uri=f"{self.path}#{r['key']}",
                                  instruction=u, output=a,
                                  metadata={"lang": r["lang"], "kind": r["kind"], "intent": r.get("intent")}))
        return out, rej


class TrimNormalizer(BaseNormalizer):
    def run(self, samples):
        for s in samples:
            s.output = re.sub(r"\s+\n", "\n", s.output.strip()).strip()
            s.output = re.sub(r"^(refusal:|assistant:|answer:)\s*", "", s.output, flags=re.I)
        return samples  # normalizers return the list itself, not a (kept, rejected) pair


class RefusalQualityGate(BaseGate):
    """Harmful samples must be an explicit, self-contained, non-leaking refusal."""

    def run(self, samples):
        kept, rej = [], []
        for s in samples:
            lang = s.metadata.get("lang", "en")
            txt = s.output
            n = len(txt.split())
            if s.metadata.get("kind") == "benign":
                if n < 15:
                    rej.append(mk_rej(s, reason="benign-too-short")); continue
                if re.search(MARKERS[lang], txt, re.I):
                    rej.append(mk_rej(s, reason="benign-contains-refusal")); continue
                kept.append(s); continue

            if n < 10:
                rej.append(mk_rej(s, reason="too-short")); continue
            if n > 110:
                rej.append(mk_rej(s, reason="too-long")); continue
            if lang == "hg" and re.search(r"[\u0900-\u097F]", txt):
                rej.append(mk_rej(s, reason="hg-written-in-devanagari")); continue
            if not re.search(MARKERS[lang], txt, re.I):
                rej.append(mk_rej(s, reason="no-explicit-refusal-marker")); continue
            low = txt.lower()
            if any(h in low for h in HOWTO):
                rej.append(mk_rej(s, reason="procedure-leak")); continue
            if re.search(r"(?m)^\s*\d+[.)]\s", txt):
                rej.append(mk_rej(s, reason="numbered-steps")); continue
            # a looping answer in the training set becomes a looping model at inference
            if loop_ratio(txt) > 0.15:
                rej.append(mk_rej(s, reason="repetitive-answer")); continue
            kept.append(s)
        return kept, rej


def loop_ratio(txt):
    """Share of repeated trigrams. A training answer that already loops teaches looping."""
    w = re.findall(r"\w+", txt.lower())
    if len(w) < 6:
        return 0.0
    t = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1.0 - len(set(t)) / len(t)


class JsonlExporter(BaseExporter):
    def __init__(self, path):
        self.path = path

    def export(self, samples, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            for s in samples:
                f.write(json.dumps({"key": s.id, "kind": s.metadata.get("kind"),
                                    "lang": s.metadata.get("lang"), "intent": s.metadata.get("intent"),
                                    "messages": [{"role": "user", "content": s.instruction},
                                                 {"role": "assistant", "content": s.output}]},
                                   ensure_ascii=False) + "\n")


def main():
    if not os.path.exists(SRC):
        print(f"missing {SRC}"); sys.exit(1)
    pipe = Pipeline(steps=[RepairReader(SRC), TrimNormalizer(), RefusalQualityGate(),
                           JsonlExporter(CURATED), MessagesExporter()], output_dir=__import__("pathlib").Path(OUTDIR))
    res = pipe.run()
    print("pipeline result:", type(res).__name__)

    rows = [json.loads(l) for l in open(CURATED)] if os.path.exists(CURATED) else []
    print(f"kept {len(rows)}")
    diag = {"kept": len(rows),
            "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in ("harmful", "benign")},
            "by_lang": {l: sum(1 for r in rows if r["lang"] == l) for l in ("en", "hi", "hg")}}
    print(json.dumps(diag, indent=2))
    json.dump(diag, open(REJECT_LOG, "w"), indent=2)

    for l in ("en", "hi", "hg"):
        ex = [r for r in rows if r["lang"] == l and r["kind"] == "harmful"]
        if ex:
            print(f"\n[{l}] {ex[0]['messages'][1]['content'][:160]}")


if __name__ == "__main__":
    main()
