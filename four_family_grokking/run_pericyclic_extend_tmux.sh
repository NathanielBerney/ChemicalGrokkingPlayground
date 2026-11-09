#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=3

/home/nate/anaconda3/envs/grokking/bin/python train_family.py \
  --family pericyclic --outdir runs/pericyclic \
  --steps 120000 --eval-every 200 --checkpoint-every 2000 \
  --train-fraction 0.30 --tolerance 0.05 --seed 0 --gpu 0 \
  --resume runs/pericyclic/last_40000.pt \
  2>&1 | tee runs/pericyclic_extend_run.log
run_status="${PIPESTATUS[0]}"
printf '\nPericyclic extension exited with status %s.\n' "$run_status"
exec bash
