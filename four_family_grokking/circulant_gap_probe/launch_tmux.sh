#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for item in raw_bits:0 log_bits:1 log_scalar:2 log_flatbits:3; do
  variant="${item%%:*}"
  gpu="${item##*:}"
  session="grok_gap_${variant}_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "$session already exists; leaving it alone"
  else
    tmux new-session -d -s "$session" "bash run_variant_tmux.sh $variant $gpu"
    echo "started $session"
  fi
done
