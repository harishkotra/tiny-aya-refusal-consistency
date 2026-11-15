#!/bin/zsh
# Curate (CuratorKIT gate) then train (LoRA repair). Sequential: one GPU job at a time.
PY=/Users/shk/.venvs/lexsi/bin/python
S=/Users/shk/.hermes/cache/scratch
D=/Volumes/Drive/lexsi/data

$PY $S/curate_repair.py                                  > $D/curate.log   2>&1; echo "curate exit=$?"
DATA=$D/repair_sft_curated.jsonl OUT=/Volumes/Drive/lexsi/out/lora_safety_v1 \
  EPOCHS=2 ACCUM=16 HF_HUB_OFFLINE=1 $PY $S/train_lora.py > $D/train.log 2>&1; echo "train exit=$?"
tail -6 $D/curate.log; tail -8 $D/train.log
