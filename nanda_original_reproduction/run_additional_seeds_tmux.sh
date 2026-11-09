#!/usr/bin/env bash
set -uo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$run_dir"
gpu="$1"
shift
export CUDA_VISIBLE_DEVICES="$gpu"

for seed in "$@"; do
  "/home/nate/anaconda3/envs/grokking/bin/python" train_modular.py \
    --seed "$seed" \
    --steps 40000 \
    --eval-every 100 \
    --checkpoint-every 2000 \
    --outdir "run_seed${seed}" \
    2>&1 | tee "run_seed${seed}_run.log"
  run_status="${PIPESTATUS[0]}"
  printf '\nSeed %s exited with status %s.\n' "$seed" "$run_status"
  if [[ "$run_status" -ne 0 ]]; then
    break
  fi
done
exec bash
