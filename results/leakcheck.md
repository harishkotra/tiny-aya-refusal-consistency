# Leak check: direct match, near-duplicate, and what it means

The evaluation prompts and the repair data come from the same taxonomy, so the risk is not a public
benchmark leak — it is **self-leak**: the model is trained on the prompts it is then evaluated on.
Checked directly, no estimation.

| check | rows | result |
|---|---|---|
| exact prompt match, eval vs SFT user turns | 288 eval rows vs 380 SFT user turns | **264 exact matches (91.7%)** |
| near-duplicate, 5-gram Jaccard ≥ 0.50 | same | 649 pairs |
| exact match, benign controls vs SFT | 120 benign rows | **116 matches (96.7%)** |
| exact match, held-out set vs SFT | 30 prompts written after training | **0** |

## The honest reading

The repair set is generated per intent: for each harmful prompt in the taxonomy, a decline is
written for *that request*. So the main table is an **in-distribution** number. It measures whether
the adapter learned to refuse these requests in the right language and register. It does not
establish generalisation, and the article states that in the summary rather than burying it.

What carries the out-of-sample claim is small and clean:

- **24 eval prompts** whose curated repair rows were rejected by CuratorKIT — base 0.083 → after
  0.958 explicit (hi 0.222 → 1.000, hg 0.000 → 0.933).
- **30 prompts written after training finished** — base 0.367 → 0.900 explicit, and hand-read
  **7/30 answering → 0/30** (`results/heldout.md`, `results/heldout_audit.md`).

Splits: 288 attack prompts + 120 benign controls, all evaluated on every arm; SFT 380 rows
(264 harmful + 116 benign), trainer's own 90/10 split inside those 380, `SEED=7`.

## Near-duplicate examples

With a 0.50 threshold the pairs are dominated by reformulations of the same intent (`direct__en__firearm`
against its own sampled rewrite at J ≈ 1.0 under stopword-stripped shingles). They are expected and
deliberate: the design goal was to teach a register for an intent so it carries to an unseen phrasing.
That goal is what the held-out set tests, and it is the only number in this repo that is free of the
self-leak.
