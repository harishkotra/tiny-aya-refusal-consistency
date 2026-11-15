"""LoRA SFT repair: teach explicit, terminating, in-language refusals.

Trains on repair_sft.jsonl (harmful -> explicit refusal, benign -> helpful answer).
Loss is masked to assistant tokens only, so the model learns the reply, not the prompt.

The failure being repaired: in romanised Hinglish the baseline almost never states a
refusal (5.2% explicit) and 15% of replies collapse into repetition loops. The target
behaviour is not a new policy, it is the policy the model already applies in English -
stated once, in the user's language, and then it stops.
"""
import json, math, os, random, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "/Volumes/Drive/lexsi/models/tiny-aya-global"
DATA = os.environ.get("DATA", "/Volumes/Drive/lexsi/data/repair_sft_curated.jsonl")
OUT = os.environ.get("OUT", "/Volumes/Drive/lexsi/out/lora_safety_v1")
MAXLEN = int(os.environ.get("MAXLEN", 640))
MB = int(os.environ.get("MB", 4))  # micro-batch: logits are [MB, T, 262144] float32, so this is the memory knob
EPOCHS = int(os.environ.get("EPOCHS", 2))
ACCUM = int(os.environ.get("ACCUM", 8))
LR = 1e-4
SEED = 7

torch.manual_seed(SEED)
random.seed(SEED)


def build_dataset(tok, path):
    samples = []
    for line in open(path):
        r = json.loads(line)
        msgs = r["messages"]
        prompt = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        aid = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        # Truncate the prompt from the left, never the answer: a row whose labels are
        # all -100 contributes no gradient and turns the batch loss into NaN.
        aid = aid[:MAXLEN - 32]
        if len(pid) + len(aid) > MAXLEN:
            pid = pid[-(MAXLEN - len(aid)):]
        ids = pid + aid
        labels = [-100] * len(pid) + aid
        if len(aid) < 5 or sum(1 for x in labels if x != -100) < 5:
            continue
        samples.append({"ids": ids, "labels": labels, "lang": r["lang"], "kind": r["kind"]})
    return samples


def make_batch(samples, pad_id):
    mx = max(len(s["ids"]) for s in samples)
    ids, labels, mask = [], [], []
    for s in samples:
        p = mx - len(s["ids"])
        ids.append([pad_id] * p + s["ids"])
        labels.append([-100] * p + s["labels"])
        mask.append([0] * p + [1] * len(s["ids"]))
    return (torch.tensor(ids), torch.tensor(labels), torch.tensor(mask))


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    data = build_dataset(tok, DATA)
    random.shuffle(data)
    n_val = max(8, int(0.06 * len(data)))
    val, train = data[:n_val], data[n_val:]
    print(f"train={len(train)} val={len(val)} per-lang:", flush=True)
    for l in ("en", "hi", "hg"):
        print(f"  {l}: {sum(1 for d in train if d['lang'] == l)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("mps")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / ACCUM)
    total = steps_per_epoch * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.1)
    print(f"steps={total} (accum={ACCUM}, epochs={EPOCHS})", flush=True)

    t0 = time.time()
    step = 0
    for ep in range(EPOCHS):
        random.shuffle(train)
        for i in range(0, len(train), ACCUM):
            chunk = train[i:i + ACCUM]
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for j in range(0, len(chunk), MB):
                sub = chunk[j:j + MB]
                ids, labels, mask = make_batch(sub, tok.pad_token_id or 0)
                ids, labels, mask = ids.to("mps"), labels.to("mps"), mask.to("mps")
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                w = len(sub) / len(chunk)
                (out.loss * w).backward()
                loss_acc += out.loss.item() * w
                del out
                torch.mps.empty_cache()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 5 == 0 or step == total:
                el = time.time() - t0
                print(f"ep{ep+1} step {step}/{total} loss={loss_acc:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"{el/step:.1f}s/step eta={(total-step)*el/step/60:.1f}m", flush=True)
            if step >= total:
                break
        # quick val loss
        model.eval()
        vl, n = 0.0, 0
        with torch.inference_mode():
            for s in val:
                ids, labels, mask = make_batch([s], tok.pad_token_id or 0)
                vl += model(input_ids=ids.to("mps"), attention_mask=mask.to("mps"),
                            labels=labels.to("mps")).loss.item()
                n += 1
        model.train()
        print(f"  == epoch {ep+1} val_loss={vl/max(1,n):.4f}", flush=True)

    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    json.dump({"base": MODEL, "r": 16, "alpha": 32, "lr": LR, "epochs": EPOCHS, "accum": ACCUM,
               "maxlen": MAXLEN, "n_train": len(train), "n_val": len(val), "seed": SEED},
              open(os.path.join(OUT, "train_config.json"), "w"), indent=2)
    print(f"\nsaved adapter -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
