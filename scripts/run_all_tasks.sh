#!/usr/bin/env bash
# Nightly batch: keep pulling tasks until there is nothing (safe) left to do.
#   run_next_task.sh exit 0 -> task done, try the next one
#   run_next_task.sh exit 3 -> idle (queue empty / stopped / quota) — normal end
#   anything else           -> needs-review or error; STOP the chain, because later
#                              tasks usually depend on this one (continuing burns tokens)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while :; do
  bash scripts/run_next_task.sh
  rc=$?
  case "${rc}" in
    0) continue ;;
    3) echo "[$(date -Is)] idle — batch finished."; exit 0 ;;
    *) echo "[$(date -Is)] stopping batch: needs-review or error (rc=${rc})."; exit 1 ;;
  esac
done
