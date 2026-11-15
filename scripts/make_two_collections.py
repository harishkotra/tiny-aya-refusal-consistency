"""Create the two collections the submission form asks for: models and datasets.

The mixed collection (4 items) stays as a landing page, but the form wants one collection
of models and one of datasets, each fully public.
"""
from huggingface_hub import HfApi
from huggingface_hub.hf_api import CollectionItem

api = HfApi()
NS = "harishkotra"

MODELS = [
    ("tiny-aya-global-refusal-consistency-lora", "model",
     "LoRA adapter (v4) that repairs in-language refusal in Tiny Aya Global."),
]
DATASETS = [
    ("tiny-aya-refusal-consistency-prompts", "dataset",
     "288 harmful prompts (8 families x 12 intents x en/hi/hg) + 120 benign controls."),
    ("tiny-aya-refusal-consistency-sft", "dataset",
     "Curated repair set (v4): 380 rows, 264 harmful + 116 benign, gated by CuratorKIT."),
    ("tiny-aya-refusal-consistency-eval", "dataset",
     "Full before/after generations from the final protocol (408 + 120 rows per arm)."),
]

def build(title, description, items):
    existing = [c for c in api.list_collections(owner=NS) if c.title == title]
    if existing:
        col = existing[0]
        print("reusing", col.slug)
    else:
        col = api.create_collection(title=title, namespace=NS, description=description, private=False)
    for repo, kind, note in items:
        api.add_collection_item(col.slug, item_id=f"{NS}/{repo}", item_type=kind, note=note,
                                exists_ok=True)
    got = api.get_collection(col.slug)
    print(title, "->", len(got.items), "items")
    return col.slug

m = build("Tiny Aya Global - refusal consistency (models)",
          "Model side: the v4 LoRA adapter. Public, un-gated.",
          MODELS)
d = build("Tiny Aya Global - refusal consistency (datasets)",
          "Data side: prompts, CuratorKIT-curated SFT set, and the full final protocol eval runs.",
          DATASETS)
print("models   ->", m)
print("datasets ->", d)
