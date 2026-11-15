# Refusal consistency across languages in Tiny Aya Global

A four-stage pipeline built on Lexsi's open safety toolkit, measuring and repairing one specific
alignment failure in a Cohere Labs model: **the model's refusals are not language-general.** In
romanised Hindi/Hinglish the same weights that refuse a harmful request in English produce
neither a refusal nor compliance — they produce degraded, looping text.

Everything here runs on a single 64 GB Apple Silicon machine. No cluster, no closed-model API in
the data path.

```
CuratorKIT  ->  LoRA repair  ->  SafeTune (SafeLoRA)  ->  eval harness
  data gate      fine-tune        post-hoc projection     metrics + audit
```

| stage | repo | what it contributes, precisely |
|---|---|---|
| data curation | [CuratorKIT](https://github.com/lexsi-ai/curatorkit) | `Pipeline` / gate contract, per-sample provenance, rejection reasons, exporter artefacts — the six gates below are CuratorKIT gates |
| repair | [SafeTune](https://github.com/lexsi-ai/safetune) | `recover.apply_safe_lora` (SafeLoRA, Hsu et al. 2024) applied to the trained adapter as a post-hoc safety projection; the supervised fine-tune itself is plain PEFT LoRA. SafeLoRA at α=0.5 left behaviour unchanged (`results/safelora.md`) |
| measurement | — | not a hackathon kit: a script-aware refusal marker, a trigram collapse rate, and a local `qwen3:4b` judge, all hand-audited. AuditKIT's loop and abstention detectors are built for agent trajectories and unanswerable-RAG cases, so they were not a fit for single-turn chat replies and are not claimed |


## The finding

288 prompts: 8 attack families × 12 harmful intents × {English, Devanagari Hindi, romanised
Hinglish}, plus a 120-prompt benign control.

| | English | Devanagari Hindi | Hinglish |
|---|---|---|---|
| explicit, in-language refusal (before) | 0.635 | 0.542 | **0.031** |
| repetition collapse (before, shared decoding control) | 0.000 | 0.000 | 0.000 |
| repetition collapse (before, no decoding control) | 0.042 | 0.031 | **0.146** |
| attack success, judged (before) | 0.000 | 0.000 | 0.000 |

The judged ASR of 0.028 in the raw final run is judge error: of 8 flagged replies, 7 are refusals
and 1 is genuine (`results/manual_audit.md`).

On this prompt set the model does not comply, in any cell. It fails *silently* in one language: no
refusal, no explanation, text shaped like an answer that is unusable as either help or safety. A
safety evaluation that only scores compliance cannot see this, and a repair that only optimises
refusal rate can hide it behind one memorised sentence.

That holds for prompts that resemble its safety training. Off that distribution it does comply: on 30
prompts written after training, the base model **answers 7** and the repaired adapter answers
**0** (`results/heldout.md`, `results/heldout_audit.md`). The 288-prompt taxonomy is 91.7% contained
in the SFT set by construction, so the in-distribution numbers are labelled as such wherever they
appear.

## The repair

380 supervised rows (harmful prompts + gated in-language refusals, benign prompts + ordinary
answers), LoRA r=16 / α=32 on 30.2M parameters (0.89%), two epochs, effective batch 16, max
length 256, ~13 minutes on an M4 Pro.

CuratorKIT gates are the part that matters. A generated refusal is rejected when it:

- names the harmful object and then explains it (echo-then-explain),
- contains a numbered procedure,
- never reaches an explicit in-language decline,
- is written in the wrong script for the cell (Hinglish rows must be Latin-only),
- loops on itself (trigram repetition > 15%),
- refuses a benign request (over-refusal contamination).

Rejections are not silent: every row carries a reason code in `curation_diagnostics.json`.

Decoding is greedy with `repetition_penalty=1.10`, applied **identically to the before and after
arms**. Without it the base model loops on 15% of Hinglish prompts and the repaired model on 43%;
with it both are 0.00. A decoder setting applied to one arm only would measure the decoder.

## Results

<!-- RESULTS_TABLE -->

| cell | explicit refusal | | collapse | | distinct replies | | judged ASR |
|---|---|---|---|---|---|---|---|
| | before | after | before | after | before | after | after |
|---|---|---|---|---|---|---|---|
| English | 0.635 | 1.000 | 0.000 | 0.000 | 0.771 | 0.969 | 0.000 |
| Devanagari Hindi | 0.542 | 0.979 | 0.000 | 0.021 | 1.000 | 0.979 | 0.000 |
| Hinglish | 0.031 | 0.948 | 0.000 | 0.000 | 1.000 | 0.990 | 0.000 |

Benign control (share of benign prompts answered with a refusal-shaped reply — the over-refusal guard):

| cell | before | after |
|---|---|---|
| English | 0.000 | 0.025 |
| Devanagari Hindi | 0.025 | 0.050 |
| Hinglish | 0.025 | 0.150 |


## Where it still fails

| family | hg explicit before | hg explicit after |
|---|---|---|
| direct | 0.00 | 1.00 |
| roleplay | 0.08 | 0.92 |
| fiction | 0.00 | 1.00 |
| authority | 0.08 | 1.00 |
| incremental | 0.00 | 1.00 |
| hypothetical | 0.00 | 1.00 |
| encoding | 0.08 | 1.00 |
| continuation | 0.00 | **0.67** |

Every residual Hinglish failure (4 of 96) is in `continuation`, where the harmful frame comes from
the prompt prefix rather than the ask — the same cell that produced the only genuine base-model
compliance in the audit. `results/report_final.md` has the full per-family table.

## Reproduction

```bash
python -m venv ~/.venvs/lexsi && source ~/.venvs/lexsi/bin/activate
pip install -r requirements.txt

export HF_HOME=/path/to/scratch/hf        # keep weights off the internal disk

python scripts/build_attackset.py         # -> attackset_v1.jsonl (288 rows)
python scripts/build_benign.py            # -> benign_v1.jsonl (120 rows)
python scripts/eval_taxonomy.py           # -> baseline_v1.jsonl
python scripts/gen_repair_data.py         # local generator (qwen3 via Ollama)
python scripts/gen_repair_v2.py           # script-pinned Hinglish pass
python scripts/gen_repair_v3.py           # diversity pass
python scripts/curate_repair.py           # CuratorKIT gates -> repair_sft_curated_v3.jsonl
python scripts/train_lora.py              # -> out/lora_safety_v4
bash  scripts/eval_final.sh               # both arms, shared decoding -> metrics_final.json
python scripts/fill_article.py            # article tables straight from the metrics file
```

Prompt sets and every raw generation are published as datasets, so each number can be recomputed
without re-running inference:

- prompts + controls: `harishkotra/tiny-aya-refusal-consistency-prompts`
- gated SFT set: `harishkotra/tiny-aya-refusal-consistency-sft`
- evaluation runs + metrics: `harishkotra/tiny-aya-refusal-consistency-eval`
- adapter: `harishkotra/tiny-aya-global-refusal-consistency-lora`

## What this is not

- Not a claim that the model is unsafe in production — a claim about *where* its safety behaviour
  fails to transfer, on one authored taxonomy.
- Not a frontier-model comparison. The closed-model line in the article is about what a stronger
  generator would have hidden, not a price-per-token estimate.
- Not a full language sweep. Three cells, one model family; the same measurement over Aya Expanse
  and more code-mixed variants is the obvious next step.

## Licence

Code MIT. Data CC-BY-4.0. Base model is Cohere Labs' `tiny-aya-global` under its own licence; the
adapter is a derivative work.
