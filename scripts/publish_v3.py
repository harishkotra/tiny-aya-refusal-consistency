"""Publish the final v3 artefacts: adapter, curated SFT set, evaluation runs.

Same target repos as before, updated in place so the collection keeps one stable set of links.
"""
import json, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi

sys.path.insert(0, "/Volumes/Drive/lexsi/repo/scripts")
from publish import (HF_USER as NS, sha, ensure, upload_dir, card_header,  # noqa
                     DS_TAXONOMY, DS_REPAIR as DS_SFT, DS_EVAL, MODEL)

D = Path("/Volumes/Drive/lexsi/data")
OUT = Path("/Volumes/Drive/lexsi/out")
TOKEN = Path("/Volumes/Drive/lexsi/hf/token").read_text().strip()
TMP = Path("/Volumes/Drive/lexsi/publish_tmp")


def fresh(sub):
    p = TMP / sub
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p


def publish_sft():
    d = fresh("sft")
    shutil.copy(D / "repair_sft_curated_v3.jsonl", d / "sft.jsonl")
    shutil.copy(D / "curation_diagnostics_v3.json", d / "curation_diagnostics.json")
    rows = [json.loads(l) for l in open(D / "repair_sft_curated_v3.jsonl")]
    from collections import Counter
    per_lang = Counter(r["lang"] for r in rows)
    per_kind = Counter(r["kind"] for r in rows)
    card_header(
        f"""license: cc-by-4.0
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

The repetition gate was added after a first repair round taught one memorised sentence
(14% distinct replies) instead of a behaviour; see the article.

Generated with a local Qwen3 model through Ollama, temperature 0.9 for the Hinglish pass,
greedy elsewhere. No closed-model API calls in the data path.
""",
        d / "README.md")
    (d / "SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.name}" for p in sorted(d.glob("*.jsonl"))) + "\n")
    r = ensure(DS_SFT); upload_dir(d, r); return r


def publish_eval():
    d = fresh("eval")
    for f, name in [("baseline_v1.jsonl", "baseline_harmful.jsonl"),
                    ("after_v3.jsonl", "repaired_harmful.jsonl"),
                    ("before_benign.jsonl", "baseline_benign.jsonl"),
                    ("after_benign_v3.jsonl", "repaired_benign.jsonl"),
                    ("probe2.jsonl", "probe_crosslingual_multiturn.jsonl"),
                    ("metrics_v3.json", "metrics.json"),
                    ("report_v3.md", "report.md")]:
        if (D / f).exists():
            shutil.copy(D / f, d / name)
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
`<|END_RESPONSE|>`. Omitting the second makes generation run to the cap, which silently
corrupts every length and truncation metric — that bug cost this project one full baseline run.
""",
        d / "README.md")
    (d / "SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.name}" for p in sorted(d.iterdir()) if p.is_file() and p.name != "SHA256SUMS") + "\n")
    r = ensure(DS_EVAL); upload_dir(d, r); return r


def publish_adapter():
    src = OUT / "lora_safety_v3"
    d = fresh("model")
    for f in src.iterdir():
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
        """# Tiny Aya Global - refusal consistency LoRA

LoRA adapter (r=16, alpha=32, 30.2M trainable parameters, 0.89% of the base) trained to make
refusals **explicit, in the language of the request, and terminating** — including romanised
Hinglish, where the base model produces neither a refusal nor an answer.

## What changes

| cell | explicit in-language refusal | repetition collapse | distinct replies |
|---|---|---|---|
| English | 0.59 -> 1.00 | 0.04 -> 0.00 | 0.94 -> 0.97 |
| Hindi (Devanagari) | 0.58 -> 0.98 | 0.03 -> 0.07 | 1.00 -> 0.89 |
| Hinglish (romanised) | 0.04 -> 0.95 | 0.15 -> 0.15 | 0.14 -> 0.90 |

Attack success is 0.00 in every cell before and after: this adapter does not make the model
safer by compliance standards, it makes the model's existing safety behaviour legible outside
English. Benign-refusal rate is flat (see the eval runs dataset).

## Training

401 rows from `harishkotra/tiny-aya-refusal-consistency-sft`; two epochs, effective batch 16,
lr 1e-4 cosine, max length 256, seed 7, on one Apple M4 Pro (MPS).

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "harishkotra/tiny-aya-global-refusal-consistency-lora")
```

## Limitations

Three language cells, one model family, an authored taxonomy. It teaches a register, not
reasoning; held-out paraphrase transfer is untested. Not a guardrail.
""",
        d / "README.md")
    r = ensure(MODEL, "model"); upload_dir(d, r, "model"); return r


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if which in ("all", "sft"):
        res["sft"] = publish_sft()
    if which in ("all", "eval"):
        res["eval"] = publish_eval()
    if which in ("all", "model"):
        res["model"] = publish_adapter()
    print(json.dumps(res, indent=1))
