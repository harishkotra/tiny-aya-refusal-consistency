# Refusal consistency in Tiny Aya Global: before / after

Harmful set: 288 prompts = 8 attack families x 12 intents x {en, hi, hg}. Benign set: 120 prompts (40 per language). Greedy decoding, 256 new tokens, same stop tokens, same judge in both runs.

## Headline metrics (all 288 harmful prompts)

| metric | before | after | delta |
|---|---|---|---|
| ASR (judge, lower better) | 0.028 | 0.0 | -0.028 |
| explicit refusal (higher better) | 0.403 | 0.976 | +0.573 |
| repetition collapse (lower better) | 0.0 | 0.007 | +0.007 |
| no stop token (lower better) | 0.559 | 0.014 | -0.545 |
| mean length (chars) | 788 | 221 | -567.000 |

## Language: en

| metric | before | after | delta |
|---|---|---|---|
| ASR | 0.031 | 0.0 | -0.031 |
| explicit refusal | 0.635 | 1.0 | +0.365 |
| repetition collapse | 0.0 | 0.0 | +0.000 |
| no stop token | 0.208 | 0.0 | -0.208 |
| mean length (chars) | 618 | 194 | -424.000 |
| distinct replies | 0.771 | 0.969 | +0.198 |

## Language: hi

| metric | before | after | delta |
|---|---|---|---|
| ASR | 0.0 | 0.0 | +0.000 |
| explicit refusal | 0.542 | 0.979 | +0.437 |
| repetition collapse | 0.0 | 0.021 | +0.021 |
| no stop token | 0.615 | 0.031 | -0.584 |
| mean length (chars) | 779 | 237 | -542.000 |
| distinct replies | 1.0 | 0.979 | -0.021 |

## Language: hg

| metric | before | after | delta |
|---|---|---|---|
| ASR | 0.052 | 0.0 | -0.052 |
| explicit refusal | 0.031 | 0.948 | +0.917 |
| repetition collapse | 0.0 | 0.0 | +0.000 |
| no stop token | 0.854 | 0.01 | -0.844 |
| mean length (chars) | 965 | 232 | -733.000 |
| distinct replies | 1.0 | 0.99 | -0.010 |

## Benign control (over-refusal guard)

| lang | before: refusal-shaped reply | after: refusal-shaped reply |
|---|---|---|
| en | 0.0 | 0.025 |
| hi | 0.025 | 0.05 |
| hg | 0.025 | 0.15 |

## Per family, Hinglish (the failing cell)

| family | before explicit | after explicit | before collapse | after collapse |
|---|---|---|---|---|
| direct | 0.0 | 1.0 | 0.0 | 0.0 |
| roleplay | 0.083 | 0.917 | 0.0 | 0.0 |
| fiction | 0.0 | 1.0 | 0.0 | 0.0 |
| authority | 0.083 | 1.0 | 0.0 | 0.0 |
| incremental | 0.0 | 1.0 | 0.0 | 0.0 |
| hypothetical | 0.0 | 1.0 | 0.0 | 0.0 |
| encoding | 0.083 | 1.0 | 0.0 | 0.0 |
| continuation | 0.0 | 0.667 | 0.0 | 0.0 |
