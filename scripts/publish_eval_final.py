"""Replace the eval dataset contents with the final protocol run (shared rep-pen 1.10).

Earlier uploads carried the v4 no-penalty run; every number in the article, the README and the
model card comes from metrics_final.json, so the dataset must match it byte for byte.
"""
import hashlib, json, os, shutil
from pathlib import Path

from huggingface_hub import HfApi

D = Path("/Volumes/Drive/lexsi/data")
OUT = Path("/Volumes/Drive/lexsi/publish_tmp/eval_final")
REPO = "harishkotra/tiny-aya-refusal-consistency-eval"


def harmful(src, dst):
    rows = [json.loads(l) for l in open(D / src)]
    rows = [r for r in rows if r.get("family") != "benign"]
    (OUT / dst).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def copy(src, dst):
    shutil.copy(D / src, OUT / dst)
    return sum(1 for _ in open(OUT / dst))


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

n = {
    "baseline_harmful.jsonl": harmful("base_f.jsonl", "baseline_harmful.jsonl"),
    "baseline_benign.jsonl": copy("base_benign_f.jsonl", "baseline_benign.jsonl"),
    "repaired_harmful.jsonl": harmful("after_f.jsonl", "repaired_harmful.jsonl"),
    "repaired_benign.jsonl": copy("after_benign_f.jsonl", "repaired_benign.jsonl"),
    "metrics.json": copy("metrics_final.json", "metrics.json"),
    "report.md": copy("report_final.md", "report.md"),
}
shutil.copy(D / "probe2.jsonl", OUT / "probe_crosslingual_multiturn.jsonl")

(OUT / "PROTOCOL.md").write_text(
    """# Evaluation protocol (final run)

Both arms — base Tiny Aya Global and the v4 adapter after the V4 patch — are decoded with:

```
do_sample=False            # greedy
max_new_tokens=256
repetition_penalty=1.10    # identical in both arms
eos_token_id=[3, 6, 261001]  # eos, <|END_OF_TURN_TOKEN|>, <|END_RESPONSE|>
```

The penalty is applied to *both* arms because it is a decoder property; applied to one arm it would
measure the decoder rather than the repair. 1.10 was selected on the Hinglish cell alone (1.05 left
a residual loop rate, 1.20 erased the refusal marker) and frozen for every cell.

Rows: 288 harmful (8 families x 12 intents x en/hi/hg), 120 benign controls (40 per language).
Every row keeps the prompt, the raw response, the judged label and the script-aware metrics, so
each number in `metrics.json` can be recomputed from the two JSONL files.
"""
)

sums = []
for p in sorted(OUT.iterdir()):
    sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
(OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")

api = HfApi()
api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(OUT), commit_message="final protocol run (shared rep-pen 1.10)")
print(json.dumps(n, indent=1))
print(f"uploaded {len(list(OUT.iterdir()))} files -> {REPO}")
