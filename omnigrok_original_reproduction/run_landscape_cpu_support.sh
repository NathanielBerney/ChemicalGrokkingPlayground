#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/logs"
declare -a pids=()
for worker in 4 5 6 7; do
    OMP_NUM_THREADS=2 "$PYTHON" "$ROOT/reproduce_qm9.py" landscape \
        --worker-index "$worker" --workers 8 --device cpu --reverse --max-jobs 30 \
        > "$ROOT/logs/landscape_cpu_support${worker}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
