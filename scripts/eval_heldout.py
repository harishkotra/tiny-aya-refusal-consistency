"""Evaluate the held-out prompt set (written after training, never in the SFT data).

Same decoding as the run of record: greedy, 256 new tokens, repetition_penalty=1.10, both arms.
No judge call — the metric here is explicit in-language refusal, which is script-aware and local.
"""
import argparse
import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/Volumes/Drive/lexsi/models/tiny-aya-global"
HELDOUT = "/Volumes/Drive/lexsi/data/heldout_v1.jsonl"
MAXNEW = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rep-pen", type=float, default=1.10)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=None, help="full model dir (merged checkpoint) instead of base+adapter")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(HELDOUT)]
    out_path = f"/Volumes/Drive/lexsi/data/{a.tag}.jsonl"

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    MODEL_DIR = a.model or MODEL
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    stops = [tok.eos_token_id, tok.convert_tokens_to_ids("<|END_RESPONSE|>")]
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).to(dev).eval()
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).eval()
        print("adapter:", a.adapter, flush=True)

    t0 = time.time()
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            p = tok.apply_chat_template([{"role": "user", "content": r["text"]}],
                                        tokenize=False, add_generation_prompt=True)
            enc = tok(p, return_tensors="pt", truncation=True, max_length=1024).to(dev)
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=stops,
                                     repetition_penalty=a.rep_pen)
            gen = out[0][enc["input_ids"].shape[1]:].tolist()
            resp = tok.decode(gen, skip_special_tokens=True).strip()
            f.write(json.dumps(dict(r, response=resp,
                                    truncated=not any(t in stops for t in gen)),
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {i+1}/{len(rows)} {(i+1)/(time.time()-t0):.2f}/s", flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
