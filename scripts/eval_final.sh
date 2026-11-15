#!/bin/bash
# Final protocol: greedy + repetition_penalty 1.10 applied IDENTICALLY to base and repaired arms.
# Rationale: the base model's romanised-Hinglish decoding collapses into repeated sentences once
# it generates at length; without a shared decoding control the before/after comparison measures
# the decoder, not the repair. 1.10 was chosen on a held-out-by-construction probe (hg subset only)
# as the smallest value that removes the loops without erasing the refusal marker.
set -u
D=/Volumes/Drive/lexsi/data; S=/Users/shk/.hermes/cache/scratch; PY=/Users/shk/.venvs/lexsi/bin/python
A=/Volumes/Drive/lexsi/out/lora_safety_v4; RP=1.10
LOG=$D/eval_final.log; : > "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

$PY $S/eval_after.py --no-adapter --rep-pen $RP --tag base_f       >> "$LOG" 2>&1
say "base harmful exit=$? rows=$(wc -l < $D/base_f.jsonl)"
$PY $S/eval_after.py --no-adapter --rep-pen $RP --only benign --tag base_benign_f >> "$LOG" 2>&1
say "base benign exit=$? rows=$(wc -l < $D/base_benign_f.jsonl)"
$PY $S/eval_after.py --adapter $A --rep-pen $RP --tag after_f      >> "$LOG" 2>&1
say "after harmful exit=$? rows=$(wc -l < $D/after_f.jsonl)"
$PY $S/eval_after.py --adapter $A --rep-pen $RP --only benign --tag after_benign_f >> "$LOG" 2>&1
say "after benign exit=$? rows=$(wc -l < $D/after_benign_f.jsonl)"

AFTER=$D/after_f.jsonl AFTER_BENIGN=$D/after_benign_f.jsonl \
BEFORE=$D/base_f.jsonl BEFORE_BENIGN=$D/base_benign_f.jsonl \
OUT_MD=$D/report_final.md OUT_JSON=$D/metrics_final.json \
  $PY $S/make_report.py >> "$LOG" 2>&1
say "report exit=$?"
say DONE
