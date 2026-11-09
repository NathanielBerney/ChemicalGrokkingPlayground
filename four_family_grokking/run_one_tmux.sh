#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p runs
family="$1"
physical_gpu="$2"
export CUDA_VISIBLE_DEVICES="$physical_gpu"

/home/nate/anaconda3/envs/grokking/bin/python train_family.py \
  --family "$family" --outdir "runs/$family" \
  --steps 40000 --eval-every 200 --checkpoint-every 2000 \
  --train-fraction 0.30 --tolerance 0.05 --seed 0 --gpu 0 \
  2>&1 | tee "runs/${family}_run.log"
run_status="${PIPESTATUS[0]}"
printf '\n%s training exited with status %s.\n' "$family" "$run_status"
exec bash
