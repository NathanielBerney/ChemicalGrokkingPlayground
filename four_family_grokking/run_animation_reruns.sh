#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python_bin=/home/nate/anaconda3/envs/grokking/bin/python
mkdir -p animations/logs

run_energy_pair() {
  local family=$1 gpu=$2
  "$python_bin" train_family.py --family "$family" --gpu "$gpu" \
    --steps 40000 --snapshot-every 1000 --outdir "animations/runs_raw/$family" \
    >"animations/logs/${family}_raw.log" 2>&1
  "$python_bin" train_family.py --family "$family" --gpu "$gpu" \
    --steps 40000 --snapshot-every 1000 --target-transform log \
    --outdir "animations/runs_log/$family" \
    >"animations/logs/${family}_log.log" 2>&1
}

run_energy_pair circulant 0 & p0=$!
run_energy_pair mobius 1 & p1=$!
run_energy_pair linear 2 & p2=$!
"$python_bin" train_family.py --family pericyclic --gpu 3 \
  --steps 120000 --snapshot-every 2000 --outdir animations/runs/pericyclic \
  >animations/logs/pericyclic.log 2>&1 & p3=$!

for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid"
done
echo "All seven archived animation reruns finished."
