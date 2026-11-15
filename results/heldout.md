# Held-out check, and the training overlap it exists to answer

## The overlap I have to declare first

`repair_sft_curated_v4.jsonl` is built per intent: for each harmful prompt in the taxonomy, the
repair generator writes a decline. So the SFT user turns **reuse the evaluation prompts**. Measured:
**264 of 288** eval prompts (91.7%) appear verbatim in the SFT set, and **116 of 120** benign
controls do (`results/leakcheck.md`, plus a 5-gram near-duplicate pass).

The headline 0.403 → 0.976 is therefore an **in-distribution** result: it measures whether the
adapter learned to refuse *these* requests in the right register. It is not a generalisation claim.
The 24 prompts curation dropped (all Hindi/Hinglish) are the only out-of-sample rows in the main
run: base 0.083 → after 0.958 (hi 0.222 → 1.000, hg 0.000 → 0.933).

## The held-out set

30 prompts written **after training finished**, unseen by construction: 10 per language, all eight
families, fresh phrasings of the same intent space (firearm, ransomware, scam, fake id, gas, ied,
spyware, laundering, pathogen, invoice fraud). Same decoding as the run of record. Scripts:
`scripts/eval_heldout.py`, `scripts/judge_heldout.py`. Raw generations:
`harishkotra/tiny-aya-refusal-consistency-eval` under `heldout/`.

| cell | n | explicit refusal, base | explicit refusal, v4 | truncation, base | truncation, v4 | mean chars, base | mean chars, v4 |
|---|---|---|---|---|---|---|---|
| English | 10 | 0.70 | **1.00** | 0.40 | 0.00 | 883 | 226 |
| Hindi (Devanagari) | 10 | 0.40 | **0.90** | 0.80 | 0.10 | 876 | 303 |
| Hinglish (romanised) | 10 | 0.00 | **0.80** | 0.90 | 0.10 | 922 | 347 |
| all | 30 | 0.367 | **0.900** | 0.70 | 0.067 | 894 | 292 |

Direction holds out of sample; the level drops, as it should. Hinglish lands at 0.80 held out versus
0.95 in-distribution, Hindi 0.90 versus 0.98.

## The number that matters more than the refusal rate

Explicit-refusal rate measures *how the model declines*. On unseen prompts the more important
question is whether it declines at all. Hand-reading all 30 in both arms (`results/heldout_audit.md`):

| | complies | partial | drift / non-answer | refuses |
|---|---|---|---|---|
| base | 4 | 3 | 3 | 20 |
| v4 | **0** | **0** | 2 | 28 |

**7 of 30 base replies answer the request or start to; none of v4's do.** The seven are all in
frames the flat taxonomy under-samples — fiction, hypothetical and roleplay, in Hindi and Hinglish —
and they include a step-by-step driver's-licence forgery guide, an IED component list, and a
finished invoice-fraud phishing email.

The local judge flagged **0 of 30** in either arm. Judged ASR on this set is 0.000 for base and v4;
hand-audited it is 0.233 and 0.000. That is a second, independent demonstration that a small local
judge under-detects caveated procedural detail, and the reason the headline metric is a script-aware
refusal marker with a hand audit on top, not a judge score.

## What this changes in the write-up

The claim is not "Tiny Aya refuses harmful requests and only the wording is wrong". It is narrower:
**on prompts that resemble its safety training it refuses, fluently or not; off that distribution it
complies, and it complies most in romanised Hinglish.** The repair works because it adds refusal
behaviour in the request's own language and register rather than teaching the model to translate a
template — which is also why the escape route is a fresh phrasing, not a different language.

> Content warning: this file and the published generations contain harmful request text and model
> output that answers it. No operational detail is reproduced beyond short quoted fragments in
> `results/heldout_audit.md`.
