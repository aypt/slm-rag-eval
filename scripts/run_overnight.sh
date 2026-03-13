#!/usr/bin/env bash
# Overnight supervisor — keeps the task chain moving WITHOUT cron. v2.
#
# What it does that a bare loop does not:
#   1. Puts the repo's .venv on PATH itself, so it survives detached shells
#      (tmux / setsid) where your interactive environment is absent.
#   2. Sleeps through provider-limit cooldowns instead of exiting — a rate limit
#      at 1am costs minutes of waiting, not the whole night. The task stays in
#      tasks/todo/ and is retried after the cooldown.
#   3. Enables one automatic fix attempt per task (FIX_ATTEMPTS=1, handled inside
#      run_next_task.sh) so a flaky first try doesn't stall the chain.
#   4. Tolerates a needs-review task and moves on to the next one, but stops
#      after MAX_CONSEC_FAILS consecutive failures (default 2) — that pattern
#      usually means the whole queue is blocked behind one broken dependency.
#   5. Prints a morning summary (done / needs-review / todo) on exit, whatever
#      the reason for exiting.
#
# IMPORTANT — set MAX_CONSEC_FAILS=1 if your task queue is a hard dependency
# chain (e.g. M03 imports the module M02 was supposed to write). Skipping ahead
# then only manufactures more failures. Check the next two task specs before
# leaving this at 2.
#
# What will still stop it — deliberately, because a human is genuinely needed:
#   - MAX_CONSEC_FAILS consecutive needs-review verdicts
#   - weekly usage reaching QUOTA_MAX_PCT (default 90)
#   - an empty queue or .agent-stop (clean exit 0)
#   - more than MAX_COOLDOWNS rate limits in a row (quota probably exhausted)
#
# Env overrides:
#   MAX_CONSEC_FAILS   consecutive needs-review verdicts tolerated (default 2)
#   MAX_COOLDOWNS      consecutive rate-limit sleeps tolerated (default 12)
#   FIX_ATTEMPTS       passed through to run_next_task.sh (default 1 here)
#   plus everything run_next_task.sh understands (CODEX_ARGS, COOLDOWN_MINUTES, ...)
#
# Usage (inside tmux):  bash scripts/run_overnight.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export FIX_ATTEMPTS="${FIX_ATTEMPTS:-1}"

MAX_CONSEC_FAILS="${MAX_CONSEC_FAILS:-2}"
MAX_COOLDOWNS="${MAX_COOLDOWNS:-12}"
FAILS=0
COOLDOWNS=0

START="$(date -Is)"
summary() {
  echo ""
  echo "==== NIGHT SUMMARY (started ${START}, ended $(date -Is)) ===="
  echo "done:";         ls tasks/done 2>/dev/null         | sed 's/^/  /'
  echo "needs-review:"; ls tasks/needs-review 2>/dev/null | sed 's/^/  /'
  echo "still todo:";   ls tasks/todo 2>/dev/null         | sed 's/^/  /'
  echo ""
  echo "next: git log --oneline main..agent/auto ; git diff main..agent/auto"
  echo "discarded work from failed attempts: logs/*-failed.patch"
  echo "============================================================"
}
trap summary EXIT

while :; do
  bash scripts/run_next_task.sh
  rc=$?

  # --- a task passed every gate --------------------------------------------
  if [[ ${rc} -eq 0 ]]; then
    FAILS=0
    COOLDOWNS=0
    continue
  fi

  # --- no verdict: cooldown, empty queue, kill switch, or quota threshold ----
  if [[ ${rc} -eq 3 ]]; then
    if [[ -f .quota-cooldown-until ]]; then
      UNTIL="$(cat .quota-cooldown-until 2>/dev/null || echo 0)"
      NOW="$(date +%s)"
      if [[ "${UNTIL}" =~ ^[0-9]+$ ]] && (( UNTIL > NOW )); then
        COOLDOWNS=$(( COOLDOWNS + 1 ))
        if (( COOLDOWNS > MAX_COOLDOWNS )); then
          echo "[$(date -Is)] ${COOLDOWNS} rate limits in a row — quota looks exhausted, stopping."
          exit 1
        fi
        SECS=$(( UNTIL - NOW + 30 ))
        echo "[$(date -Is)] provider limit (${COOLDOWNS}/${MAX_COOLDOWNS}) — sleeping $(( SECS / 60 + 1 )) min, then resuming."
        sleep "${SECS}"
        continue
      fi
    fi
    echo "[$(date -Is)] idle (queue empty / .agent-stop / quota threshold) — stopping cleanly."
    exit 0
  fi

  # --- a task was filed to needs-review -------------------------------------
  FAILS=$(( FAILS + 1 ))
  sleep 10   # guards against a tight spin if the runner ever fails before archiving
  if (( FAILS >= MAX_CONSEC_FAILS )); then
    echo "[$(date -Is)] ${FAILS} consecutive failures — stopping; the queue is probably blocked."
    exit 1
  fi
  echo "[$(date -Is)] task needs review (${FAILS}/${MAX_CONSEC_FAILS}) — tree rolled back, moving to the next task."
done
