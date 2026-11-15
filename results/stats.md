# Statistics

Recomputed from the published rows (`harishkotra/tiny-aya-refusal-consistency-eval`),
not from the summary JSON. Paired bootstrap over prompts, 4,000 resamples, seed 11.

| language | n | explicit before | explicit after | delta | 95% CI (paired bootstrap) |
|---|---|---|---|---|---|
| en | 96 | 0.635 | 1.000 | +0.365 | [+0.271, +0.458] |
| hi | 96 | 0.542 | 0.979 | +0.438 | [+0.344, +0.542] |
| hg | 96 | 0.031 | 0.948 | +0.917 | [+0.854, +0.969] |
| all | 288 | 0.403 | 0.976 | +0.573 | [+0.517, +0.632] |

Discordant pairs: 165 prompts gained an explicit refusal, 0 lost one (of 288). Under a sign test that is p ≈ 4.28e-50 (two-sided, exact binomial).

## Judge check

The judge flagged **8** of 288 base replies as compliant. I read all 8 by hand: **7 decline the request, 1 complies** (a `continuation` prompt that pre-fills a "practical guide"). So the judge's positive class is wrong 7 times out of 8 — precision 12% — and judged ASR 0.028 is really 1/288 = 0.003.

Only 1 of those 8 carries an *explicit* refusal marker, which is the point of the headline metric: these are refusals delivered in a softer register than the regex looks for. The marked-refusal rate is therefore a **lower bound** on refusal behaviour, never an overcount — it under-reports before the repair, so the 0.03 Hinglish figure is if anything flattering to the base model, not harsh.

Of the 280 replies the judge passed, 115 carry a refusal marker: the judge does not miss refusals, it reads them as compliance.

A Cohen's κ would need a labelled random sample; only the positive class was audited, so none is claimed. After the repair the judge flagged **0 of 288**, and the four residual Hinglish failures were read by hand as well.

## Throughput (basis for the cost estimate)

Wall clock from the run of record, on an M4 Pro (16-core GPU, 64 GB), fp16 on MPS, greedy,
micro-batch 1 — no serving framework, no batching, one process:

- `base benign` — 120 replies in 238 s = **1.98 s/reply**
- `after harmful` — 408 replies in 790 s = **1.94 s/reply**
- `after benign` — 120 replies in 276 s = **2.30 s/reply**

Mean **2.07 s/reply**, i.e. **~35 min per 1,000 replies** for prompts of this shape.

After-arm output: 63,626 chars / 10,963 words over 288 replies (221 chars per reply) ≈ **61 output tokens per reply**, so **~61k output tokens per 1,000 replies**. Measured throughput ≈ **30 tok/s**, decoding-bound, as expected on MPS at micro-batch 1.

## Cost per 1,000 replies

| option | time | est. cost / 1k replies | note |
|---|---|---|---|
| this machine (M4 Pro, local) | 35 min | ~$0.02 electricity | 65 W under load, no API meter, prompts never leave the box |
| same model on a rented M4-class box | 35 min | ~$0.30 | at ~$0.50/h rental, the honest price of self-hosting this throughput |
| gpt-4o-mini, same prompts | ~5 s | ~$0.04 | 40k in + 61k out at $0.15/$0.60 per M tokens |
| gpt-4.1 / 4o class | ~10 s | ~$1.00 | 40k in + 61k out at $2.00/$8.00 per M tokens |

Reading: at micro-batch 1 on a laptop-class GPU this is roughly **7× the price of a small closed model** and 8× faster than nothing but slower than any hosted endpoint. The case for it is not cost — it is that a 3.4 B multilingual model that refuses in the language it was asked in can run where the data cannot leave, and that a 121 MB adapter is cheaper to distribute than a fine-tune. Batching (vLLM, batch 16 at 187 tok/s measured earlier) would cut the self-hosted number by ~6×, to roughly $0.05/1k.
