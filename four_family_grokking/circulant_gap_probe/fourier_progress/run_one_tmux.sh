#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
variant="$1"
physical_gpu="$2"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
mkdir -p "runs/$variant"
extra=()
if [[ "$variant" != "log_scalar" ]]; then
  extra+=(--cosine-decay)
fi
/home/nate/anaconda3/envs/grokking/bin/python ../train_gap.py \
  --variant "$variant" --outdir "runs/$variant" --steps 40000 \
  --eval-every 200 --checkpoint-every 2000 --archive-every 1000 \
  --seed 0 --gpu 0 "${extra[@]}" \
  2>&1 | tee "runs/${variant}.log"
run_status="${PIPESTATUS[0]}"
printf '\n%s Fourier-progress training exited with status %s.\n' "$variant" "$run_status"
exec bash
