#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES=0

"/home/nate/anaconda3/envs/grokking/bin/python" grok_circulant_n42.py \
  --steps 40000 \
  --eval-every 200 \
  --checkpoint-every 2000 \
  --outdir circulant_grokking_runs/seed0_tmux \
  2>&1 | tee circulant_grokking_runs/seed0_tmux_run.log
run_status="${PIPESTATUS[0]}"
printf '\nTraining process exited with status %s.\n' "$run_status"
exec bash
