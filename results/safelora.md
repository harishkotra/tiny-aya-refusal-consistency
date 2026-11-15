# SafeTune stage: SafeLoRA projection of the v4 adapter

SafeLoRA (Hsu et al., NeurIPS 2024, arXiv:2405.16833) is training-free: it projects the fine-tune
delta onto the subspace spanned by the safety-relevant weights of the aligned model, on the theory
that safety capability is a small number of directions that fine-tuning pushes away from.

Call, verbatim from `scripts/safelora.py`:

```python
proj = apply_safe_lora(merged, aligned_state_dict=BASE_SD, base_state_dict=BASE_SD, alpha=0.5)
```

The library normally wants *base* (pre-alignment) and *aligned* weights as separate checkpoints.
Here the fine-tune starts from the aligned release, so both are the same state dict — that
substitution is the one deviation from the paper's setting, and it means "safety subspace" here is
measured against the model's own released weights rather than against a pretrain checkpoint.

## What it did

| | |
|---|---|
| tensors changed vs base | 253 of 290 |
| mean abs change | 4.7e-05 |
| max abs change | 9.0e-04 |
| held-out explicit refusal, adapter | 0.90 (en 1.00 / hi 0.90 / hg 0.80) |
| held-out explicit refusal, + SafeLoRA | **0.90 (en 1.00 / hi 0.90 / hg 0.80)** |
| surviving failures | the same three rows, in the same order |

Behaviour is identical on all 30 held-out prompts; the projection moved weights by a thousandth and
moved nothing measurable.

## Reading

Not a null result for SafeLoRA in general — a null result for SafeLoRA *here*. The projection exists
to strip deltas that degrade safety. This adapter's delta does the opposite: it adds refusal
behaviour, and the baseline it was trained from is the aligned model, so there is no unsafe
direction for the projection to find. Reported because it is a cheap, honest falsification attempt:
if the repair had been quietly damaging safety while raising refusal rate, this stage was the test
that would have shown it, and it did not.

Cost: 4 minutes, one forward pass over the merged checkpoint, no training.
