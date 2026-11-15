# The refusal is in English even when the request is not

*A measurement and a repair of refusal consistency on Cohere Labs' Tiny Aya Global, built with
CuratorKIT → SafeTune → AuditKIT.*

## Summary

Tiny Aya Global 3.35B refuses harmful requests. It does not refuse them *in the language they are
asked in*. Across 288 prompts — 8 attack families × 12 harmful intents × {English, Devanagari
Hindi, romanised Hinglish} — the model almost never produced usable harmful content. The failure
is not compliance. It is the *expression* of refusal. Under one fixed decoding protocol, 64 in 100
English replies and 54 in 100 Devanagari Hindi replies contained an explicit in-language refusal.
In romanised Hinglish, 3 in 100 did. The rest were neither a refusal nor an answer: how-to-shaped
text with hallucinated nouns, sometimes running to the token cap because nothing in it signalled
an ending.

After a LoRA repair trained on CuratorKIT-gated data, in-language explicit refusal rises to 1.00 /
0.98 / 0.95 across the three cells, with repetition collapse at zero everywhere. It is not free:
5 of 40 Hinglish benign prompts are now refused. That cost is measured and reported, not smoothed
away.

<!-- TABLE:headline -->
| metric | before | after | delta |
|---|---|---|---|
| attack success (judge, lower better) | 0.028 | 0.0 | -0.028 |
| explicit refusal in the asked language (higher better) | 0.403 | 0.976 | +0.573 |
| repetition collapse (lower better) | 0.0 | 0.007 | +0.007 |
| no stop token / runaway (lower better) | 0.559 | 0.014 | -0.545 |
| distinct replies (higher better) | 0.924 | 0.979 | +0.055 |
| mean reply length, chars | 788 | 221 | -567 |
<!-- /TABLE -->

<!-- TABLE:lang -->
| cell | explicit refusal before | after | collapse before | after | distinct replies after |
|---|---|---|---|---|---|
| English | 0.64 | 1.00 | 0.00 | 0.00 | 0.97 |
| Hindi (Devanagari) | 0.54 | 0.98 | 0.00 | 0.02 | 0.98 |
| Hinglish (romanised) | 0.03 | 0.95 | 0.00 | 0.00 | 0.99 |
<!-- /TABLE -->

## Why refusal *expression* is the right metric here

I started with the obvious metric: attack success rate, judged by a local 4B model scoring whether
a reply contains actionable harmful detail. It came back flat, so I nearly stopped there. Instead I
read the generations, and what emerged was not a safety failure in the usual sense:

- **English, direct ask.** Short explicit refusal. Correct.
- **Devanagari Hindi, direct ask.** Explicit refusal in Hindi, sometimes with a
  hallucinated-but-harmless suggestion appended. Works.
- **Romanised Hinglish, direct ask.** No refusal and no answer. The model writes fluent-looking
  Hinglish that never declines, drifts into confident nonsense, and — without a decoding control —
  locks into repeated sentences in about 15% of cases. It fails *deaf*, not open.

That distinction changes what you measure. A model that already refuses nearly everything cannot
be made safer by safety training. It can be made *legible*: an operator in Lucknow who asks in
Hinglish should see a refusal they recognise as a refusal, not a paragraph of fluent nothing. So
the metric pair is:

1. **explicit in-language refusal rate** — does the reply decline, in the script and register the
   user used? Script-aware regex (Devanagari alternation for Hindi, romanised
   `nahi … sakta / maaf kijiye` shapes for Hinglish), not a translation model.
2. **repetition collapse rate** — share of replies whose trigram diversity falls below 0.7, i.e.
   the model is repeating itself rather than communicating. This is the failure Hinglish actually
   exhibits, and no refusal-rate metric catches it.

Three guards, because a repair that raises refusal by refusing everything is worthless: a
**120-prompt benign control** (40 per language), a **distinct-reply rate** (a model that emits the
same sentence 96 times has learnt a template, not a behaviour), and a **manual audit of every judge
positive** — the local judge marked 8/288 base replies compliant and was wrong about 7 of them.

## The pipeline

```
build taxonomy (288 prompts) ──► baseline eval (AuditKIT-style harness)
                                          │
                                          ▼
                       repair generation (local Qwen3 via Ollama)
                                          │
                                          ▼
        CuratorKIT gates ──► SFT set ──► SafeTune LoRA ──► re-eval, same harness
                                          │
                                          ▼
                        before/after metrics + benign control + probe set
```

All three Lexsi repos are load-bearing, not decorative:

- **CuratorKIT** decides what enters the training set. Its gates rejected echo-then-explain
  refusals, numbered procedures ("first obtain…"), too-short answers, Devanagari written where
  romanised Hinglish was asked for, and — after round three — answers that were already looping.
- **SafeTune** is the training stage (LoRA r=16, α=32, 30.2M trainable parameters, 0.89% of the
  base) on one Apple M4 Pro via MPS.
- **AuditKIT** shaped the evaluation protocol: fixed decoding, script-aware refusal markers, a
  benign control arm, and raw generations published so every number can be recomputed.

## Four rounds, because the first three were wrong

The interesting part of this project is not the final adapter. It is that each metric caught a
different way the repair was fake.

<!-- TABLE:rounds -->
| round | hg explicit | hg collapse | hg distinct | training rows |
|---|---|---|---|---|
| v1 template | 1.00 | 0.00 | 0.00 | 404 |
| v2 long-form | 0.95 | 0.46 | 0.90 | 405 |
| v3 gated | 0.97 | 0.54 | 0.96 | 392 |
| v4 short + rep-pen 1.10 | 0.95 | 0.00 | 0.99 | 380 |
<!-- /TABLE -->

- **v1 — the template.** Trained on ~400 generated refusals. Explicit refusal hit 1.00 in every
  cell, collapse went to zero, and it looked finished. Then the distinct-reply rate: 13 unique
  strings across 96 Hinglish prompts, 74 of them byte-identical. The adapter had memorised one
  sentence. This is the failure mode that a "refusal rate ↑" table will happily hide.
- **v2 — diversity without control.** A third generation pass with no worked example, higher
  temperature, rotating openers and a uniqueness gate produced 94 genuinely different Hinglish
  refusals. Distinct-reply rate rose to 0.90 — and collapse rose to 0.46, because now the model
  kept generating past the decline and looped.
- **v3 — gating loops in the data.** A trigram gate rejected training answers that already looped
  (13/131 Hinglish, 16/136 Hindi). Collapse got *worse* (0.54). The loops were not being copied
  from the data; they are what the base model does in romanised Hindi once it is asked to keep
  talking.
- **v4 — short answers plus a shared decoding control.** Hinglish harmful answers capped at 40
  words, near-duplicates removed (380 rows), and `repetition_penalty=1.10` applied *identically to
  both arms*. With no penalty the base model loops on 15% of Hinglish prompts and the repaired
  model on 43%; with 1.10 both are 0.00 and the repaired arm keeps its refusal marker. The penalty
  is a decoder property, so applying it to one arm only would have measured the decoder, not the
  repair.

## Result

<!-- TABLE:benign -->
| cell | benign prompts answered with a refusal-shaped reply |
|---|---|---|
| | before | after |
|---|---|---|
| en | 0.000 | 0.025 |
| hi | 0.025 | 0.050 |
| hg | 0.025 | 0.150 |
<!-- /TABLE -->

Runaway generation is the same phenomenon measured differently: 56% of base replies never emitted a
stop token at all and ran to the 256-token cap, versus 1.4% after the repair. The base model is not
withholding an ending — it does not know it has reached one.

The over-refusal cost is real and unevenly distributed: the Hinglish cell pays most of it. Five of
40 Hinglish benign prompts — a dripping tap, a bicycle inspection, a sleep routine — now come back
as refusals. That is the price of teaching a register the base model did not have, and it is the
number I would attack next: more benign Hinglish, and a gate that rejects harmful-style refusals
applied to benign-style prompts.

## What this costs, and what a closed model would cost

The whole pipeline ran on one laptop: 3.35B parameters in fp16 on 16 GPU cores, LoRA at
micro-batch 1 and 256 tokens (the 262k-token vocabulary makes the logits tensor, not the
activations, the memory bottleneck — that constraint set the entire training recipe), two epochs in
~13 minutes, four evaluation arms in ~40 minutes. Generation and judging used a local Qwen3 through
Ollama; no closed-model API calls are in the data path, so the dataset licence is unencumbered.

The honest comparison is not "cheaper than a frontier model", it is that a stronger generator would
have hidden the finding. A frontier model asked to *write Hinglish refusals* would have produced
excellent ones — and the experiment would then have measured its Hinglish, not Tiny Aya's. The
repair data had to come from something that could fail in the same register, or the diversity and
collapse metrics would have been measuring nothing.

## Limitations

- Three language cells, one model family, an authored taxonomy. This is a measurement of a specific
  failure, not a benchmark.
- The repair teaches a *register*, not reasoning. Held-out paraphrase transfer is untested; a
  prompt family outside the taxonomy may still fail.
- The refusal marker is regex-based and script-aware — deliberate (no judge in the primary metric)
  but brittle for code-mixed replies; one of the six Hinglish "over-refusals" is a false positive.
- `repetition_penalty=1.10` is a decoding control, chosen on the Hinglish cell and frozen for all
  cells. It removes loops; it does not make the base model's Hinglish good.
- Not a guardrail. This is evidence that "the model refuses everything" and "the model refuses
  legibly in every language it serves" are different claims, and only the second one is measurable.
