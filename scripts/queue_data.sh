#!/bin/zsh
# Sequential data build: no two jobs contend for the same Ollama instance.
set -x
PY=/Users/shk/.venvs/lexsi/bin/python
S=/Users/shk/.hermes/cache/scratch
D=/Volumes/Drive/lexsi/data

$PY $S/build_benign.py      > $D/benign_run.log    2>&1; echo "benign exit=$?"
$PY $S/gen_repair_data.py   > $D/repair_gen.log    2>&1; echo "repair_gen exit=$?"
tail -4 $D/benign_run.log; tail -4 $D/repair_gen.log
