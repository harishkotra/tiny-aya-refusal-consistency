#!/bin/bash
# v3: re-curate v2 data with the repetition gate, retrain, re-eval. Same recipe otherwise.
set -u
D=/Volumes/Drive/lexsi/data; S=/Users/shk/.hermes/cache/scratch; PY=/Users/shk/.venvs/lexsi/bin/python
LOG=$D/queue_v3.log; : > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

CURATED=$D/repair_sft_curated_v3.jsonl OUTDIR=$D/curated_v3 \
REJECT_LOG=$D/curation_diagnostics_v3.json \
  $PY $S/curate_repair.py >> "$LOG" 2>&1
say "curate exit=$? rows=$(wc -l < $D/repair_sft_curated_v3.jsonl)"
python3 - "$D" <<'EOF' >> "$LOG" 2>&1
import json, sys, collections
d = sys.argv[1]
rows = [json.loads(l) for l in open(f"{d}/repair_sft_curated_v3.jsonl")]
print("per-lang:", collections.Counter(r["lang"] for r in rows))
EOF

DATA=$D/repair_sft_curated_v3.jsonl OUT=/Volumes/Drive/lexsi/out/lora_safety_v3 \
EPOCHS=2 ACCUM=16 MB=1 MAXLEN=256 HF_HUB_OFFLINE=1 \
  $PY $S/train_lora.py >> "$LOG" 2>&1
say "train exit=$?"

A=/Volumes/Drive/lexsi/out/lora_safety_v3
$PY $S/eval_after.py --adapter "$A" --tag after_v3 >> "$LOG" 2>&1
say "eval harmful exit=$? rows=$(wc -l < $D/after_v3.jsonl)"
$PY $S/eval_after.py --only benign --adapter "$A" --tag after_benign_v3 >> "$LOG" 2>&1
say "eval benign exit=$? rows=$(wc -l < $D/after_benign_v3.jsonl)"

AFTER=$D/after_v3.jsonl AFTER_BENIGN=$D/after_benign_v3.jsonl \
BEFORE=$D/baseline_v1.jsonl BEFORE_BENIGN=$D/before_benign.jsonl \
OUT_MD=$D/report_v3.md OUT_JSON=$D/metrics_v3.json \
  $PY $S/make_report.py >> "$LOG" 2>&1
say "report exit=$?"
say DONE
