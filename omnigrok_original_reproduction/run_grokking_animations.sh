#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/logs"

declare -a pids=()
for spec in "100 3 0" "100 1 1" "400 3 2" "400 1 3"; do
    read -r total_size alpha gpu <<< "$spec"
    OMP_NUM_THREADS=2 "$PYTHON" "$ROOT/reproduce_qm9.py" grokking \
        --total-size "$total_size" --alpha "$alpha" --device "cuda:$gpu" \
        --run-suffix animation --archive-every-steps 1000 \
        > "$ROOT/logs/grokking_total${total_size}_alpha${alpha}_animation.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
