"""SafeTune stage: SafeLoRA projection of the v4 adapter onto the base model's safety subspace.

SafeLoRA (Hsu et al., NeurIPS 2024, arXiv:2405.16833) is training-free: it projects the fine-tune
delta onto the subspace spanned by the safety-relevant weights of the aligned model. Here the
"aligned" model is the released Tiny Aya Global chat model and the fine-tune is the v4 adapter.

Writes a merged, projected checkpoint so the eval script can treat it as a plain model dir.

Note: SafeLoRA normally takes base (pre-alignment) and aligned weights separately. Here the
fine-tune starts from the aligned release, so both paths are the same checkpoint.
"""
import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from safetune.recover import apply_safe_lora

BASE = "/Volumes/Drive/lexsi/models/tiny-aya-global"
ADAPTER = "/Volumes/Drive/lexsi/out/lora_safety_v4"
OUT = "/Volumes/Drive/lexsi/out/safelora_v4"

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER)
merged = model.merge_and_unload().eval()
print("merged", sum(p.numel() for p in merged.parameters()) / 1e9, "B params", flush=True)

from pathlib import Path

from safetensors.torch import load_file

BASE_SD = {}
for shard in sorted(Path(BASE).glob("*.safetensors")):
    BASE_SD.update(load_file(shard, device="cpu"))
print("base state dict", len(BASE_SD), "tensors", flush=True)

proj = apply_safe_lora(merged, aligned_state_dict=BASE_SD, base_state_dict=BASE_SD, alpha=0.5)
print("projected", flush=True)

sd = {k: v.detach().to("cpu", copy=True) for k, v in proj.state_dict().items()}
save_file(sd, f"{OUT}/model.safetensors")
print("saved", f"{OUT}/model.safetensors", flush=True)
