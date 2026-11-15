#!/bin/bash
# v2 chain: wait for the diverse Hinglish rewrites, then curate -> train -> eval -> report.
# Everything is serialised because the 14B generator, the trainer and the eval all compete
# for the same 64 GB of unified memory.
set -u
D=/Volumes/Drive/lexsi/data
S=/Users/shk/.hermes/cache/scratch
PY=/Users/shk/.venvs/lexsi/bin/python
LOG=$D/queue_v2.log
: > "$LOG"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# 1. wait for generation to finish (file stable + no writer process)
say "waiting for hg diversity pass"
while pgrep -f gen_repair_v3.py >/dev/null; do sleep 20; done
say "hg rows: $(wc -l < $D/repair_hg_diverse.jsonl)"

# 2. curate with the new hinglish answers taking precedence
CURATED=$D/repair_sft_curated_v2.jsonl OUTDIR=$D/curated_v2 \
REJECT_LOG=$D/curation_diagnostics_v2.json \
  $PY $S/curate_repair.py >> "$LOG" 2>&1
say "curate exit=$? rows=$(wc -l < $D/repair_sft_curated_v2.jsonl)"
grep -E 'kept|reject' "$LOG" | tail -12

# 3. train
DATA=$D/repair_sft_curated_v2.jsonl OUT=/Volumes/Drive/lexsi/out/lora_safety_v2 \
EPOCHS=2 ACCUM=16 MB=1 MAXLEN=256 HF_HUB_OFFLINE=1 \
  $PY $S/train_lora.py >> "$LOG" 2>&1
say "train exit=$?"

# 4. eval: harmful + benign, adapter on
$PY $S/eval_after.py --adapter /Volumes/Drive/lexsi/out/lora_safety_v2 --tag after_v2 >> "$LOG" 2>&1
say "eval harmful exit=$?"
$PY $S/eval_after.py --only benign --adapter /Volumes/Drive/lexsi/out/lora_safety_v2 --tag after_benign_v2 >> "$LOG" 2>&1
say "eval benign exit=$?"

# 5. tables
AFTER=$D/after_v2.jsonl AFTER_BENIGN=$D/after_benign_v2.jsonl \
BEFORE=$D/baseline_v1.jsonl BEFORE_BENIGN=$D/before_benign.jsonl \
OUT_MD=$D/report_v2.md OUT_JSON=$D/metrics_v2.json \
  $PY $S/make_report.py >> "$LOG" 2>&1
say "report exit=$?"
say "DONE"
