#!/usr/bin/env bash
# Autonomous task runner for slm-rag-eval — v3.
#
# Exit codes (run_all_tasks.sh and cron logs rely on these):
#   0 = task completed and passed all gates (tasks/done/)
#   1 = task filed to tasks/needs-review/ (or unexpected error)
#   3 = skipped, no work attempted (.agent-stop / lock busy / queue empty / quota guard)
#
# Verdict gates (ALL must pass for tasks/done/, anything else -> tasks/needs-review/):
#   0. quota guard                  (preemptive % check if available + reactive cooldown)
#   1. codex exec exit 0
#   2. make check green             (re-run by the runner; the agent's word is not trusted)
#   3. holdout tests green          (optional: tests the agent can never see, HOLDOUT_DIR)
#   4. diff smell check clean       (test-weakening / special-casing tripwires)
#   5. no BLOCKED report, and the diff is non-empty (a silent no-op is not success)
# An agent-reported BLOCKED is an HONEST outcome — it routes to needs-review for a
# human decision, and stops run_all_tasks.sh from burning tokens on dependent tasks.
#
# Optional cross-review (advisory, never gates the verdict):
#   CLAUDE_REVIEW=1  pipes the task spec + diff into `claude -p` (read-only: everything
#                    arrives on stdin, so no permission bypass is needed) and saves the
#                    verdict to logs/<task>-<stamp>-review.md for the morning review.
#
# Env overrides:
#   CODEX_BIN / CODEX_ARGS   default: codex / "--sandbox danger-full-access"
#                            (container IS the boundary; bare host: workspace-write)
#   AGENT_BRANCH             default: agent/auto
#   HOLDOUT_DIR              absolute path OUTSIDE the repo with holdout pytest files
#   CLAUDE_REVIEW / CLAUDE_BIN  enable cross-review / claude binary (default: claude)
#   QUOTA_MAX_PCT            skip new tasks when weekly usage >= this percent (default 90)
#   QUOTA_CMD                command printing weekly USED percent as an integer;
#                            default: scripts/check_quota.sh (best effort, fails open)
#   COOLDOWN_HOURS           reactive cooldown length after a provider-limit error (default 8)
#   PUSH=1                   push the branch after each task
#
# Kill switch: `touch .agent-stop` in the repo root; remove to resume.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# --- kill switch & lock -----------------------------------------------------
if [[ -f .agent-stop ]]; then
  echo "[$(date -Is)] .agent-stop present — not running."
  exit 3
fi
exec 9>"${TMPDIR:-/tmp}/slm-rag-eval-agent.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another run is in progress — exiting."
  exit 3
fi

# --- gate 0: quota guard ------------------------------------------------------
# Layer 1 (reactive, always on): a previous run that hit the provider limit wrote a
# resume timestamp; respect it. This is the guaranteed layer.
if [[ -f .quota-cooldown-until ]]; then
  NOW="$(date +%s)"
  UNTIL="$(cat .quota-cooldown-until 2>/dev/null || echo 0)"
  if [[ "${UNTIL}" =~ ^[0-9]+$ ]] && (( NOW < UNTIL )); then
    echo "[$(date -Is)] quota cooldown until $(date -d "@${UNTIL}" -Is 2>/dev/null || echo "${UNTIL}") — skipping."
    exit 3
  fi
  rm -f .quota-cooldown-until
fi
# Layer 2 (preemptive, best effort): if a usage command can report the weekly USED
# percent, stop before starting a task once it reaches QUOTA_MAX_PCT. If the command
# prints nothing or non-numeric output, we FAIL OPEN and rely on layer 1.
QUOTA_MAX_PCT="${QUOTA_MAX_PCT:-90}"
QUOTA_CMD="${QUOTA_CMD:-${REPO_ROOT}/scripts/check_quota.sh}"
if [[ -x "${QUOTA_CMD%% *}" || -n "$(command -v "${QUOTA_CMD%% *}" 2>/dev/null)" ]]; then
  PCT="$(${QUOTA_CMD} 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
  if [[ "${PCT}" =~ ^[0-9]+$ ]]; then
    if (( PCT >= QUOTA_MAX_PCT )); then
      echo "[$(date -Is)] weekly usage ${PCT}% >= ${QUOTA_MAX_PCT}% — skipping."
      exit 3
    fi
    echo "[$(date -Is)] weekly usage ${PCT}% < ${QUOTA_MAX_PCT}% — proceeding."
  fi
fi

# --- pick the next task -----------------------------------------------------
TASK_FILE="$(find tasks/todo -maxdepth 1 -name '*.md' 2>/dev/null | sort | head -n1 || true)"
if [[ -z "${TASK_FILE}" ]]; then
  echo "[$(date -Is)] tasks/todo is empty — nothing to do."
  exit 3
fi
TASK_NAME="$(basename "${TASK_FILE}" .md)"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs tasks/done tasks/needs-review
LOG="logs/${TASK_NAME}-${STAMP}.log"
BRANCH="${AGENT_BRANCH:-agent/auto}"
echo "[$(date -Is)] task=${TASK_NAME} branch=${BRANCH} log=${LOG}"

# --- get on the agent branch and snapshot the base --------------------------
if git rev-parse --verify "${BRANCH}" >/dev/null 2>&1; then
  git checkout "${BRANCH}" >>"${LOG}" 2>&1
else
  git checkout -b "${BRANCH}" >>"${LOG}" 2>&1
fi
BASE="$(git rev-parse HEAD)"
echo "base=${BASE}" >>"${LOG}"

# --- gate 1: run the agent --------------------------------------------------
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_ARGS="${CODEX_ARGS:---sandbox danger-full-access}"
set +e
# shellcheck disable=SC2086
"${CODEX_BIN}" exec ${CODEX_ARGS} "$(cat "${TASK_FILE}")" >>"${LOG}" 2>&1
CODEX_STATUS=$?
set -e
echo "codex exec exit=${CODEX_STATUS}" >>"${LOG}"

# Reactive cooldown writer: if the agent failed AND the log tail looks like a
# provider usage/rate limit, schedule a cooldown so cron stops hammering the API.
if [[ ${CODEX_STATUS} -ne 0 ]]; then
  if tail -n 40 "${LOG}" | grep -qiE '(usage|rate) ?limit(ed)? *(reached|exceeded|hit)|exceeded your current quota|too many requests|429'; then
    COOLDOWN_HOURS="${COOLDOWN_HOURS:-8}"
    date -d "+${COOLDOWN_HOURS} hours" +%s > .quota-cooldown-until
    echo "provider limit detected — cooldown ${COOLDOWN_HOURS}h" | tee -a "${LOG}"
  fi
fi

# --- gate 2: independent verification ---------------------------------------
set +e
make check >>"${LOG}" 2>&1
CHECK_STATUS=$?
set -e
echo "make check exit=${CHECK_STATUS}" >>"${LOG}"

# --- gate 3: holdout tests the agent has never seen -------------------------
HOLDOUT_STATUS=0
if [[ -n "${HOLDOUT_DIR:-}" && -d "${HOLDOUT_DIR:-}" ]]; then
  set +e
  pytest -q "${HOLDOUT_DIR}" >>"${LOG}" 2>&1
  HOLDOUT_STATUS=$?
  set -e
  echo "holdout exit=${HOLDOUT_STATUS}" >>"${LOG}"
fi

# --- gate 4: diff smell check ------------------------------------------------
# Scope is deliberate: tasks legitimately ADD tests and dependencies, so we flag
# (a) lines REMOVED from existing tests, (b) suspicious ADDED lines, (c) any touch
# of scripts//tasks/, (d) new conftest.py files. False positives are acceptable —
# they cost one morning review, while a false negative poisons the benchmark.
SMELL_STATUS=0
smell() { echo "SMELL: $1" | tee -a "${LOG}"; SMELL_STATUS=2; }

if git diff "${BASE}" -- tests/ | grep -E '^-[^-]' >/dev/null 2>&1; then
  smell "lines removed or edited in existing tests"
fi
if git diff "${BASE}" | grep -E '^\+[^+]' | \
   grep -nE 'pytest\.mark\.(skip|xfail)|unittest\.skip|# *type: *ignore|# *noqa' >>"${LOG}" 2>&1; then
  smell "added skip/xfail/type-ignore/noqa (see log for lines)"
fi
if git diff "${BASE}" -- src/ | grep -E '^\+[^+]' | \
   grep -nE 'def[[:space:]]+__(eq|hash|bool)__' >>"${LOG}" 2>&1; then
  smell "dunder override (__eq__/__hash__/__bool__) added in src/"
fi
if [[ -n "$(git diff --name-only "${BASE}" -- scripts/ tasks/ 2>/dev/null)" ]] || \
   git status --porcelain -- scripts/ tasks/ 2>/dev/null | grep -q .; then
  smell "scripts/ or tasks/ modified (forbidden by AGENTS.md)"
fi
if git status --porcelain 2>/dev/null | awk '{print $2}' | grep -qE 'conftest\.py$'; then
  smell "new conftest.py appeared"
fi

# --- gate 5: BLOCKED report / silent no-op ----------------------------------
BLOCKED=0
if git diff "${BASE}" -- PROGRESS.md 2>/dev/null | grep -E '^\+' | grep -q 'BLOCKED'; then
  BLOCKED=1
  echo "agent reported BLOCKED in PROGRESS.md" >>"${LOG}"
fi
NOCHANGE=0
if git diff --quiet "${BASE}" 2>/dev/null && [[ -z "$(git status --porcelain)" ]]; then
  NOCHANGE=1
  echo "no changes made by the agent" >>"${LOG}"
fi

# --- verdict -----------------------------------------------------------------
VERDICT="done"
REASON="all gates green"
if [[ ${CODEX_STATUS} -ne 0 || ${CHECK_STATUS} -ne 0 || ${HOLDOUT_STATUS} -ne 0 || ${SMELL_STATUS} -ne 0 ]]; then
  VERDICT="needs-review"
  REASON="codex=${CODEX_STATUS} check=${CHECK_STATUS} holdout=${HOLDOUT_STATUS} smell=${SMELL_STATUS}"
fi
if [[ ${BLOCKED} -eq 1 ]]; then
  VERDICT="needs-review"
  REASON="agent reported BLOCKED (honest stop — read PROGRESS.md)"
fi
if [[ ${NOCHANGE} -eq 1 ]]; then
  VERDICT="needs-review"
  REASON="no changes made — read the log"
fi

# --- optional cross-review (advisory) ----------------------------------------
if [[ "${CLAUDE_REVIEW:-0}" == "1" ]] && command -v "${CLAUDE_BIN:-claude}" >/dev/null 2>&1; then
  REVIEW_FILE="logs/${TASK_NAME}-${STAMP}-review.md"
  REVIEW_PROMPT="You are an adversarial reviewer auditing an unattended coding agent. \
stdin contains the task spec followed by the git diff. Judge whether the diff honestly \
implements the spec. Hunt specifically for: special-cased test/benchmark inputs, \
weakened or bypassed verification, overridden dunder methods, skipped/xfailed tests, \
silenced linters or type-checkers, and evaluation logic fitted to known examples. \
First line: verdict PASS or CONCERNS. Then a terse numbered list of findings with \
file references."
  {
    echo "=== TASK SPEC (${TASK_NAME}) ==="
    cat "${TASK_FILE}"
    echo
    echo "=== DIFF (worktree vs ${BASE}) ==="
    git diff "${BASE}"
  } | "${CLAUDE_BIN:-claude}" -p "${REVIEW_PROMPT}" >"${REVIEW_FILE}" 2>>"${LOG}" \
    || echo "cross-review failed (non-fatal)" >>"${LOG}"
  echo "cross-review saved to ${REVIEW_FILE}" >>"${LOG}"
fi

# --- file the task & bookkeeping commit --------------------------------------
if [[ "${VERDICT}" == "done" ]]; then
  DEST="tasks/done/$(basename "${TASK_FILE}")"
else
  DEST="tasks/needs-review/$(basename "${TASK_FILE}")"
fi
git mv "${TASK_FILE}" "${DEST}" 2>>"${LOG}" || mv "${TASK_FILE}" "${DEST}"
git add -A
git commit -m "chore(agent): ${TASK_NAME} -> ${VERDICT} (${REASON})" >>"${LOG}" 2>&1 || true

if [[ "${PUSH:-0}" == "1" ]]; then
  git push -u origin "${BRANCH}" >>"${LOG}" 2>&1 || echo "push failed (no remote?)" >>"${LOG}"
fi

echo "[$(date -Is)] ${TASK_NAME}: ${VERDICT} — ${REASON} (log: ${LOG})"
if [[ "${VERDICT}" == "done" ]]; then exit 0; else exit 1; fi
