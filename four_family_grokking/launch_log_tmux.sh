#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
families=(circulant mobius linear)
for gpu in 0 1 2; do
  family="${families[$gpu]}"
  session="grok_${family}_log_gpu${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    printf '%s already exists\n' "$session"
    continue
  fi
  tmux new-session -d -s "$session" "bash $(pwd)/run_log_tmux.sh $family $gpu"
  printf 'Started %s on GPU %s\n' "$session" "$gpu"
done
