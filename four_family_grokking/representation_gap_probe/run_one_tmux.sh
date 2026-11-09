#!/usr/bin/env bash
set -euo pipefail
kind="$1"
gpu="$2"
steps="${3:-2000}"
run_set="${4:-runs_seed0}"
seed="${5:-0}"
outdir="$(dirname "$0")/$run_set/$kind"
mkdir -p "$outdir"
export CUDA_VISIBLE_DEVICES="$gpu"
exec /home/nate/anaconda3/envs/grokking/bin/python -u "$(dirname "$0")/train.py" \
  --kind "$kind" --outdir "$outdir" --steps "$steps" --eval-every 250 --seed "$seed" \
  > "$outdir/train.log" 2>&1
