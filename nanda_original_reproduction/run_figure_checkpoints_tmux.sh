#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0

/home/nate/anaconda3/envs/grokking/bin/python train_modular.py \
  --seed 0 \
  --steps 40000 \
  --eval-every 100 \
  --checkpoint-every 2000 \
  --archive-every 1000 \
  --outdir figure_run_seed0 \
  2>&1 | tee figure_run_seed0_run.log
run_status="${PIPESTATUS[0]}"
printf '\nFigure checkpoint run exited with status %s.\n' "$run_status"
exec bash
