#!/usr/bin/env bash
# Supervise a pretraining run on an interruptible instance.
#
# scripts/train.py exits 75 after a preemption signal, having written a
# checkpoint. This relaunches it so the run continues from that checkpoint.
# Any other non-zero exit is a real failure and stops the supervisor.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_ROOT/configs/train.yaml}"
MAX_RESTARTS="${MAX_RESTARTS:-100}"
PREEMPTED_EXIT_CODE=75

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

attempt=0
while :; do
  attempt=$((attempt + 1))
  printf '\n=== attempt %d/%d: %s\n' "$attempt" "$MAX_RESTARTS" "$CONFIG"

  python3 "$REPO_ROOT/scripts/train.py" --config "$CONFIG"
  status=$?

  if [ "$status" -eq 0 ]; then
    echo "Run finished."
    exit 0
  fi
  if [ "$status" -ne "$PREEMPTED_EXIT_CODE" ]; then
    echo "Training failed with exit code $status; not restarting." >&2
    exit "$status"
  fi
  if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
    echo "Preempted $attempt times; giving up." >&2
    exit "$status"
  fi

  echo "Preempted. Resuming from the newest checkpoint."
  sleep 5
done
