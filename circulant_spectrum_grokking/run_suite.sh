#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${SPECTRUM_PYTHON:-/home/nate/anaconda3/envs/grokking/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg
stage="${1:-all}"
mkdir -p logs runs
pids=()
gpu=0
for seed in 0 1; do
  for target in sorted mode; do
    name="${target}_seed${seed}"
    if [[ "$stage" == train || "$stage" == all ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" train.py --target "$target" --seed "$seed" \
        --steps 40000 --outdir "runs/$name" >"logs/$name.log" 2>&1 &
      pids+=("$!")
    elif [[ "$stage" == analyze ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" analyze.py --run "runs/$name" \
        >"logs/${name}_analysis.log" 2>&1 &
      pids+=("$!")
    else
      echo 'Usage: bash run_suite.sh [all|train|analyze]' >&2; exit 2
    fi
    gpu=$((gpu+1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then echo 'A job failed; inspect logs/' >&2; exit 1; fi
if [[ "$stage" == all ]]; then bash run_suite.sh analyze; fi
if [[ "$stage" == analyze ]]; then "$PYTHON" gallery.py; fi
