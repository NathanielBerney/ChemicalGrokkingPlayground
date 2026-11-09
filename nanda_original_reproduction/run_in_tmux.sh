#!/usr/bin/env bash
set -uo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$run_dir"
export CUDA_VISIBLE_DEVICES=1

"/home/nate/anaconda3/envs/grokking/bin/python" train_modular.py \
  --steps 40000 \
  --eval-every 100 \
  --checkpoint-every 2000 \
  --outdir run_seed0 \
  2>&1 | tee run_seed0_run.log
run_status="${PIPESTATUS[0]}"
printf '\nTraining process exited with status %s.\n' "$run_status"
exec bash
