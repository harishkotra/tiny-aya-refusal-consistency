# Refusal consistency in Tiny Aya Global: before / after

Harmful set: 288 prompts = 8 attack families x 12 intents x {en, hi, hg}. Benign set: 120 prompts (40 per language). Greedy decoding, 256 new tokens, same stop tokens, same judge in both runs.

## Headline metrics (all 288 harmful prompts)

| metric | before | after | delta |
|---|---|---|---|
| ASR (judge, lower better) | 0.0 | 0.003 | +0.003 |
| explicit refusal (higher better) | 0.406 | 0.979 | +0.573 |
| repetition collapse (lower better) | 0.073 | 0.191 | +0.118 |
| no stop token (lower better) | 0.635 | 0.139 | -0.496 |
| mean length (chars) | 799 | 320 | -479.000 |

## Language: en

| metric | before | after | delta |
|---|---|---|---|
| ASR | 0.0 | 0.0 | +0.000 |
| explicit refusal | 0.594 | 1.0 | +0.406 |
| repetition collapse | 0.042 | 0.0 | -0.042 |
| no stop token | 0.375 | 0.0 | -0.375 |
| mean length (chars) | 680 | 205 | -475.000 |
| distinct replies | 0.938 | 0.99 | +0.052 |

## Language: hi

| metric | before | after | delta |
|---|---|---|---|
| ASR | 0.0 | 0.01 | +0.010 |
| explicit refusal | 0.583 | 0.979 | +0.396 |
| repetition collapse | 0.031 | 0.146 | +0.115 |
| no stop token | 0.677 | 0.104 | -0.573 |
| mean length (chars) | 811 | 299 | -512.000 |
| distinct replies | 1.0 | 0.938 | -0.062 |

## Language: hg

| metric | before | after | delta |
|---|---|---|---|
| ASR | 0.0 | 0.0 | +0.000 |
| explicit refusal | 0.042 | 0.958 | +0.916 |
| repetition collapse | 0.146 | 0.427 | +0.281 |
| no stop token | 0.854 | 0.312 | -0.542 |
| mean length (chars) | 907 | 457 | -450.000 |
| distinct replies | 1.0 | 0.948 | -0.052 |

## Benign control (over-refusal guard)

| lang | before: refusal-shaped reply | after: refusal-shaped reply |
|---|---|---|
| en | 0.0 | 0.0 |
| hi | 0.025 | 0.05 |
| hg | 0.0 | 0.05 |

## Per family, Hinglish (the failing cell)

| family | before explicit | after explicit | before collapse | after collapse |
|---|---|---|---|---|
| direct | 0.0 | 1.0 | 0.167 | 0.333 |
| roleplay | 0.0 | 1.0 | 0.167 | 0.583 |
| fiction | 0.0 | 1.0 | 0.0 | 0.167 |
| authority | 0.25 | 1.0 | 0.417 | 0.583 |
| incremental | 0.0 | 1.0 | 0.167 | 0.5 |
| hypothetical | 0.0 | 1.0 | 0.083 | 0.583 |
| encoding | 0.083 | 1.0 | 0.083 | 0.333 |
| continuation | 0.0 | 0.667 | 0.083 | 0.333 |
