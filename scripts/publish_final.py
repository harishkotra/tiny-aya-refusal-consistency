"""Publish the final artefacts (adapter, SFT set, eval runs) selected by env vars.

V=4            adapter dir suffix and data file suffix
METRICS=...    metrics json used to write the model card table
"""
import json, os, shutil, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/Volumes/Drive/lexsi/repo/scripts")
from publish import (HF_USER as NS, sha, ensure, upload_dir, card_header,  # noqa
                     DS_REPAIR as DS_SFT, DS_EVAL, MODEL)

D = Path("/Volumes/Drive/lexsi/data")
OUT = Path("/Volumes/Drive/lexsi/out")
V = os.environ.get("V", "4")
METRICS = Path(os.environ.get("METRICS", D / f"metrics_v{V}.json"))
TMP = Path("/Volumes/Drive/lexsi/publish_tmp")
NAMES = {"en": "English", "hi": "Hindi (Devanagari)", "hg": "Hinglish (romanised)"}


def fresh(sub):
    p = TMP / sub
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p


def card_table(M):
    L = ["| cell | explicit in-language refusal | repetition collapse | distinct replies |",
         "|---|---|---|---|"]
    for k in ("en", "hi", "hg"):
        b, a = M["harmful"]["before"]["by_lang"][k], M["harmful"]["after"]["by_lang"][k]
        L.append(f"| {NAMES[k]} | {b['explicit']:.2f} -> {a['explicit']:.2f} | "
                 f"{b['degenerate']:.2f} -> {a['degenerate']:.2f} | "
                 f"{b.get('unique', 1):.2f} -> {a.get('unique', 1):.2f} |")
    return "\n".join(L)


def main():
    M = json.load(open(METRICS))
    sft_src = D / f"repair_sft_curated_v{V}.jsonl"
    diag_src = D / f"curation_diagnostics_v{V}.json"
    if not diag_src.exists():
        diag_src = D / "curation_diagnostics_v3.json"
    rows = [json.loads(l) for l in open(sft_src)]
    per_lang, per_kind = Counter(r["lang"] for r in rows), Counter(r["kind"] for r in rows)
    res = {}

    # 1. SFT set
    d = fresh("sft")
    shutil.copy(sft_src, d / "sft.jsonl")
    shutil.copy(diag_src, d / "curation_diagnostics.json")
    card_header(
        """license: cc-by-4.0
language:
- en
- hi
tags:
- hinglish
- safety
- sft
- refusal
pretty_name: Tiny Aya refusal-consistency SFT set
""",
        f"""# Repair set: refusals that are explicit, in-language, and terminating

{len(rows)} rows that survived CuratorKIT's gates — {per_kind['harmful']} harmful prompts paired with
an explicit refusal in the language of the prompt, {per_kind['benign']} benign prompts paired with an
ordinary helpful answer (the over-refusal control). Per language: en {per_lang['en']}, hi
{per_lang['hi']}, hg {per_lang['hg']}.

## Fields

| field | meaning |
|---|---|
| `key` | prompt id, matches the prompts dataset |
| `lang` | `en`, `hi` (Devanagari), `hg` (romanised Hinglish) |
| `kind` | `harmful` or `benign` |
| `intent` | harmful intent label (null for benign) |
| `messages` | user turn + assistant answer |

## Gates applied (rejections in `curation_diagnostics.json`, with reason codes)

- `no-explicit-refusal-marker` — the answer advises without declining
- `procedure-leak`, `numbered-steps` — a refusal that still explains how
- `hg-written-in-devanagari` — wrong script for the cell
- `repetitive-answer` — the answer itself loops (trigrams repeat beyond 15%)
- `benign-contains-refusal` — over-refusal contamination of the training set
- `too-short`, `too-long`

The repetition gate and the length cap were added after earlier rounds taught a memorised
sentence (14% distinct replies) and then long rambling refusals (54% loop rate); both rounds are
reported in the article rather than quietly replaced.

Generated with a local Qwen3 model via Ollama (temperature 0.9 for the Hinglish pass, greedy
elsewhere). No closed-model API calls in the data path.
""",
        d / "README.md")
    (d / "SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.name}" for p in sorted(d.glob("*.jsonl"))) + "\n")
    res["sft"] = ensure(DS_SFT); upload_dir(d, res["sft"])

    # 2. eval runs
    d = fresh("eval")
    files = [("baseline_v1.jsonl", "baseline_harmful.jsonl"),
             (f"after_v{V}.jsonl", "repaired_harmful.jsonl"),
             ("before_benign.jsonl", "baseline_benign.jsonl"),
             (f"after_benign_v{V}.jsonl", "repaired_benign.jsonl"),
             ("probe2.jsonl", "probe_crosslingual_multiturn.jsonl"),
             (f"metrics_v{V}.json", "metrics.json"),
             (f"report_v{V}.md", "report.md")]
    for a, b in files:
        if (D / a).exists():
            shutil.copy(D / a, d / b)
    card_header(
        """license: cc-by-4.0
language:
- en
- hi
tags:
- hinglish
- safety
- evaluation
- refusal
pretty_name: Tiny Aya refusal-consistency evaluation runs
""",
        """# Evaluation runs: before and after the repair

Raw generations for the same prompts under the same decoding parameters, so every number in
the article can be recomputed without re-running generation.

| file | contents |
|---|---|
| `baseline_harmful.jsonl` | base model, 288 harmful prompts |
| `repaired_harmful.jsonl` | LoRA-repaired model, same 288 prompts |
| `baseline_benign.jsonl` | base model, 120 benign prompts (over-refusal control) |
| `repaired_benign.jsonl` | repaired model, same 120 benign prompts |
| `probe_crosslingual_multiturn.jsonl` | cross-lingual and four-turn escalation probes |
| `metrics.json` | computed metrics, machine-readable |
| `report.md` | the same metrics as the published tables |

Per row: prompt, response, `judge_compliant` (attack-success label), language, family, intent.

Decoding: greedy, 256 new tokens, `eos_token_id` includes both `<|END_OF_TURN_TOKEN|>` and
`<|END_RESPONSE|>`. Omitting the second makes generation run to the cap, which silently corrupts
every length and truncation metric — that bug cost this project one full baseline run.
""",
        d / "README.md")
    (d / "SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.name}" for p in sorted(d.iterdir())
                                           if p.is_file() and p.name != "SHA256SUMS") + "\n")
    res["eval"] = ensure(DS_EVAL); upload_dir(d, res["eval"])

    # 3. adapter
    d = fresh("model")
    for f in (OUT / f"lora_safety_v{V}").iterdir():
        if f.is_file():
            shutil.copy(f, d / f.name)
    card_header(
        """base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language:
- en
- hi
tags:
- hinglish
- safety
- lora
- refusal
pretty_name: Tiny Aya Global - refusal consistency LoRA
""",
        f"""# Tiny Aya Global - refusal consistency LoRA

LoRA adapter (r=16, alpha=32, 30.2M trainable parameters, 0.89% of the base) trained to make
refusals **explicit, in the language of the request, and terminating** — including romanised
Hinglish, where the base model produces neither a refusal nor an answer.

## What changes

{card_table(M)}

Attack success does not move in a way that matters: the base model's judged rate of 0.028 is
almost entirely judge error (8 replies flagged, 7 audited as refusals, 1 genuine — a
`continuation` prompt that pre-fills a "practical guide"), so hand-audited ASR is 1/288 before and
0/288 after. This adapter does not make the model safer by compliance standards; it makes its
existing safety behaviour legible outside English.

That repair is not free. Benign over-refusal rises from 0.00/0.03/0.03 to 0.03/0.05/0.15
(en/hi/hg) — five of forty Hinglish benign prompts (a dripping tap, a bicycle check, a sleep
routine) now come back as refusals. Reported, not smoothed over; `results/manual_audit.md` in the
code repo has the row-level audit.

Decoding: greedy with `repetition_penalty=1.10`, **identical in both arms** — without it the base
model loops on 15% of Hinglish prompts and the repaired model on 43%, so applying it to one arm
would measure the decoder rather than the repair.

Code and article: https://github.com/harishkotra/tiny-aya-refusal-consistency · collection:
https://huggingface.co/collections/harishkotra/tiny-aya-global-refusal-consistency-across-scripts-6af9007d3ce7e09c334db33a

## Training

{len(rows)} rows from `{NS}/tiny-aya-refusal-consistency-sft`; two epochs, effective batch 16,
lr 1e-4 cosine, max length 256, seed 7, one Apple M4 Pro (MPS).

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "{MODEL}")
```

## Limitations

Three language cells, one model family, an authored taxonomy. It teaches a register, not
reasoning; held-out paraphrase transfer is untested. Not a guardrail.
""",
        d / "README.md")
    res["model"] = ensure(MODEL, "model"); upload_dir(d, res["model"], "model")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
