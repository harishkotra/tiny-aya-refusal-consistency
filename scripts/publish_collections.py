"""Create the HF collection and attach every artefact.

The hackathon requires a public *collection* of models and datasets, not just loose repos.
This is idempotent: re-running adds missing items and skips the ones already in place.
"""
import json, sys
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = Path("/Volumes/Drive/lexsi/hf/token").read_text().strip()
NS = "harishkotra"
TITLE = "Tiny Aya Global - refusal consistency across scripts"

DESC = (
    "Refusal consistency across scripts in Tiny Aya Global: attack success ~0, but romanised Hinglish barely refuses. CuratorKIT -> SafeTune -> AuditKIT."
)



ITEMS = [
    ("model", "tiny-aya-global-refusal-consistency-lora",
     "LoRA r=16 adapter, 30.2M trainable (0.89%)"),
    ("dataset", "tiny-aya-refusal-consistency-prompts",
     "288 harmful prompts (8 families x 12 intents x en/hi/hg) + 120 benign controls"),
    ("dataset", "tiny-aya-refusal-consistency-sft",
     "404 CuratorKIT-gated repair rows + rejection diagnostics"),
    ("dataset", "tiny-aya-refusal-consistency-eval",
     "Raw generations, metrics.json and the before/after tables"),
]


def main():
    api = HfApi(token=TOKEN)
    coll = api.create_collection(title=TITLE, namespace=NS, description=DESC, private=False,
                                 exists_ok=True)
    existing = {(getattr(i, 'type', getattr(i, 'repo_type', '?')), i.id) for i in coll.items}
    print("collection:", coll.slug, "| items:", len(existing))

    for kind, name, note in ITEMS:
        item_id = f"{NS}/{name}"
        if (kind, item_id) in existing:
            print("  have:", item_id)
            continue
        api.add_collection_item(collection_slug=coll.slug, item_id=item_id,
                                item_type=kind, note=note)
        print("  added:", item_id)

    final = api.get_collection(coll.slug)
    print(json.dumps({"url": f"https://huggingface.co/collections/{final.slug}",
                      "items": [f"{getattr(i, 'type', getattr(i, 'repo_type', '?'))}:{i.id}" for i in final.items]}, indent=1))


if __name__ == "__main__":
    main()
