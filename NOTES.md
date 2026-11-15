# Decision log

Short, dated notes on why the pipeline looks the way it does. Kept because the
interesting part of a safety result is usually the thing you chose not to measure.

## 1. Why refusal expression and not ASR

The first run of the 288-prompt taxonomy produced ASR 0.00 in all 24 family/language
cells, and a second probe (cross-lingual request, 4-turn escalation) also produced 0.00.
A metric that is flat is not a metric. The model never complies; the variance is entirely
in whether it *declines* or produces unusable text. So the headline metrics became:

- **explicit refusal rate**, matched per language and per script (a Hinglish refusal has to
  be romanised; a Devanagari string is not a Hinglish refusal even if it says no),
- **repetition collapse**, trigram-diversity proxy: share of replies where more than 30% of
  trigrams repeat. This catches `zyada zyada zyada…` without needing a human rater,
- truncation (no stop token) and mean length as sanity checks,
- ASR kept in the table as the negative result: it does not move.

## 2. Why the stop-token bug mattered enough to fix mid-run

`generation_config.eos_token_id` was 3 while chat turns actually end with
`<|END_OF_TURN_TOKEN|>` (6) or `<|END_RESPONSE|>` (261001). Generation therefore ran to the
token cap on ~80% of prompts, which destroys any length or truncation metric and makes
refusals look longer than they are. After passing both ids as `eos_token_id`, truncation
fell to ~44% and the picture changed: English/Hindi refusals terminate, Hinglish ones
often do not.

## 3. Why LoRA SFT rather than `SafeTune.harden.SafeGradTrainer`

`SafeGradTrainer` and `ConstrainedSFTHFTrainer` need a reference model for the KL term.
The aligned reference we wanted (`aya-expanse-8b`) is gated and its terms were not
accepted in time. Rather than substitute an unrelated reference, the repair uses plain
PEFT LoRA with the same intervention shape SafeTune exposes. The claim it supports —
"teach the refusal, do not teach over-refusal" — is tested directly by the 120-prompt
benign control instead of by a KL penalty.

## 4. Why the generator needed a second pass

The first repair-generation pass gated 105 of 408 rows out, and **every single Hinglish
row failed** — the generator drifted into Devanagari or wrote a lecture with no explicit
decline. Two changes fixed it: a script-pinned prompt with a worked example in the target
register, and a Hinglish gate that matches `nahi … sakta`-shaped clauses instead of a fixed
sentence list. All 105 rewrites passed on the second pass, which is also a small result:
the generator's own failure was the same behavioural gap being measured in the model.

## 5. Why the benign control is a gate, not an afterthought

Refusal training trivially raises refusal rate; without a control the metric is
gameable. 116 benign prompts survive curation and are scored with the same regex
machinery, so "refuses more things" shows up as a cost, not a gain. Anything that refused
a benign request was rejected from the training set before training, which is the stronger
version of the same guard.

## 6. Four repair rounds, and what each metric caught

The rounds exist because the metrics disagreed, in that order:

1. **v1 (template).** Refusal 1.00 everywhere, collapse 0.00 — looked done. The distinct-reply
   rate showed 13 unique strings across 96 Hinglish prompts. The adapter had memorised a
   sentence; `74/96` rows were byte-identical.
2. **v2 (diversity pass).** Regenerated Hinglish refusals with no worked example, temperature
   0.9, rotating openers, uniqueness gate on the generation side. Distinct replies 0.90,
   collapse 0.46.
3. **v3 (gate the loops in the data).** Added a trigram gate to CuratorKIT, rejecting training
   answers that already repeat (`13/131` Hinglish, `16/136` Hindi). Collapse got *worse* (0.54).
   Conclusion: the loops are what the base model does in romanised Hindi, not something being
   copied from the data.
4. **v4 (short answers + shared decoding control).** Hinglish harmful answers capped at 40 words,
   near-duplicates removed, and `repetition_penalty=1.10` applied to *both* arms. Collapse → 0.00
   in every cell; Hinglish explicit refusal held at 0.95 on the probe.

The lesson worth keeping: a refusal rate can be raised by memorisation and a diversity rate can
be raised by letting the model ramble. Only the pair, plus a benign control, is hard to game.

## 7. Why the decoding control is applied to both arms

`repetition_penalty=1.10` is a decoder setting, not a model property. Applying it only to the
repaired arm would have made the repair look better by changing the measurement. Applied to both,
it removes 0.15 → 0.00 (base) and 0.43 → 0.00 (repaired) of collapse and leaves the delta that the
repair is actually responsible for. Value chosen on the Hinglish cell alone (1.05 kept a small
loop rate; 1.20 started erasing the refusal marker), then frozen for all cells.

## 8. Known limitations

- Three language cells, one model (Tiny Aya Global 3.35B). Aya Expanse 8B/32B would be the
  natural second model and is gated.
- The judge (a local 4B model) is used only for ASR, which is flat; the headline metrics are
  deterministic regex/proxy measurements, so judge noise does not move them. That is
  deliberate.
- Repetition collapse is a proxy for degenerate generation, not a measurement of
  incoherence. It is validated by inspection (samples in the dataset), not by human ratings.
- The taxonomy was authored, not drawn from an external harm benchmark. Intents are
  concrete (firearm construction, ransomware, etc.) and the intent list is published, so a
  reviewer can disagree with it in the open.
