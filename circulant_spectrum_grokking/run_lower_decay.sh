#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${SPECTRUM_PYTHON:-/home/nate/anaconda3/envs/grokking/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg
mkdir -p logs runs
pids=()
for seed in 0 1; do
  name="sorted_seed${seed}_wd0p1"
  CUDA_VISIBLE_DEVICES="$((seed*2))" "$PYTHON" train.py --target sorted --seed "$seed" \
    --steps 40000 --weight-decay .1 --outdir "runs/$name" >"logs/$name.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
