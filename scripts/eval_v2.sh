#!/bin/bash
# v2 eval only (curation + training already finished): harmful, benign, then tables.
set -u
D=/Volumes/Drive/lexsi/data; S=/Users/shk/.hermes/cache/scratch; PY=/Users/shk/.venvs/lexsi/bin/python
A=/Volumes/Drive/lexsi/out/lora_safety_v2
LOG=$D/eval_v2.log; : > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

test -f "$A/adapter_model.safetensors" || { say "no adapter at $A"; exit 1; }
say "adapter: $A"

$PY $S/eval_after.py --adapter "$A" --tag after_v2 >> "$LOG" 2>&1
say "harmful exit=$? rows=$(wc -l < $D/after_v2.jsonl)"
$PY $S/eval_after.py --only benign --adapter "$A" --tag after_benign_v2 >> "$LOG" 2>&1
say "benign exit=$? rows=$(wc -l < $D/after_benign_v2.jsonl)"

AFTER=$D/after_v2.jsonl AFTER_BENIGN=$D/after_benign_v2.jsonl \
BEFORE=$D/baseline_v1.jsonl BEFORE_BENIGN=$D/before_benign.jsonl \
OUT_MD=$D/report_v2.md OUT_JSON=$D/metrics_v2.json \
  $PY $S/make_report.py >> "$LOG" 2>&1
say "report exit=$?"
say DONE
