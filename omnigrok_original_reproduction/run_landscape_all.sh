#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/logs"
phase="${1:-all}"
if [[ "$phase" != "all" && "$phase" != "cpu" && "$phase" != "gpu" ]]; then
    printf 'Usage: %s [all|cpu|gpu]\n' "$0" >&2
    exit 2
fi

run_phase() {
    local mode="$1"
    local cuts="$2"
    local -a pids=()
    local worker gpu
    if [[ "$mode" == "cpu" ]]; then
        for worker in 0 1 2 3; do
            OMP_NUM_THREADS=2 "$PYTHON" "$ROOT/reproduce_qm9.py" landscape \
                --worker-index "$worker" --workers 8 --device cpu $cuts \
                > "$ROOT/logs/landscape_${mode}_${worker}${cuts:+_cuts}.log" 2>&1 &
            pids+=("$!")
        done
    else
        for worker in 4 5 6 7; do
            gpu=$((worker - 4))
            OMP_NUM_THREADS=2 "$PYTHON" "$ROOT/reproduce_qm9.py" landscape \
                --worker-index "$worker" --workers 8 --device "cuda:$gpu" $cuts \
                > "$ROOT/logs/landscape_${mode}_${worker}${cuts:+_cuts}.log" 2>&1 &
            pids+=("$!")
        done
    fi
    local status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    return "$status"
}

if [[ "$phase" == "all" ]]; then
    run_phase cpu "" &
    cpu_pid=$!
    run_phase gpu "" &
    gpu_pid=$!
    wait "$cpu_pid"
    wait "$gpu_pid"
    run_phase cpu "--cuts" &
    cpu_pid=$!
    run_phase gpu "--cuts" &
    gpu_pid=$!
    wait "$cpu_pid"
    wait "$gpu_pid"
else
    run_phase "$phase" ""
    run_phase "$phase" "--cuts"
fi
