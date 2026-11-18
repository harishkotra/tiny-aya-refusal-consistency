# Launch posts

Two ready-to-post versions. Images live in `assets/`.

## LinkedIn

Tiny Aya Global 3.35B refuses harmful requests. It just doesn't refuse them in the language you asked in.

I measured it across 288 prompts — 8 attack styles (roleplay, fiction, authority, continuation, encoding…) × 12 harmful intents × English, Devanagari Hindi and romanised Hinglish — holding decoding fixed.

Explicit refusal, in the language asked:
English 64%
Hindi 54%
Hinglish 3%

Same content. Same weights. Only the script changed.

Then I repaired it: 380 CuratorKIT-gated rows, one LoRA (r=16, 30M params, 13 minutes on an M4 Pro). Hinglish goes 3% → 95%, and runaway generation — replies that never emit a stop token — 56% → 1.4%.

The part I'd ask a reviewer to read first, though, is the negative result. On 30 prompts written after training, the base model still answered 7: a forgery guide, an IED parts list, a finished phishing email, two firearms walkthroughs. A local 4B judge caught none of the 7. The refusal rates above are 91.7% in-distribution, and the honest claim is narrower than it first looks — this model doesn't refuse harmful requests, it refuses requests that resemble its safety training.

Code → github.com/harishkotra/tiny-aya-refusal-consistency
Models + datasets → huggingface.co/harishkotra (collections)
Write-up → repo article.md

Built solo for the Lexsi Labs AI Safety & Alignment Hackathon, Track 2a + 2b. Thanks @Lexsi Labs — CuratorKIT did real work here (392 → 380 rows, per-row reject reasons), and SafeTune's SafeLoRA projection came back a clean null, which is now in the write-up as one.

## X (thread)

1.
Tiny Aya 3.35B refuses harmful requests — just not in the language you ask in.

288 prompts, 8 attack styles × 12 intents × EN / HI / romanised Hinglish.

Explicit in-language refusal: EN 64%, HI 54%, Hinglish 3%.

After a 13-min LoRA: 3% → 95%.

2.
Same content, same weights — only the script changes. Devanagari Hindi holds up; romanised Hinglish collapses. A model can read the risk and still fail to *say* no in the register it was asked.

3.
The honest part: on 30 prompts written after training, the base model still answered 7 (forgery guide, IED parts, a finished phishing email). A 4B judge caught 0 of 7.

Base 7/30 → repaired 0/30.

The refusal rates are in-distribution. The narrow claim is the true one.

4.
Built for the @Lexsi_labs AI Safety hackathon, Track 2a+2b.

Code: github.com/harishkotra/tiny-aya-refusal-consistency
Models + data: huggingface.co/harishkotra
Write-up: repo article.md

CuratorKIT did the gating; SafeTune's SafeLoRA was a clean null result, reported as one.

## Images

- `assets/refusal_by_language.png` — the finding. Lead with it on both platforms.
- `assets/indist_vs_unseen.png` — the honesty beat. Attach to the LinkedIn post second; on X, put it on tweet 3.
