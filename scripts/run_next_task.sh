#!/usr/bin/env bash
# Autonomous task runner for slm-rag-eval — v3.3.
#
# Exit codes (run_overnight.sh relies on these):
#   0 = task completed and passed all gates (filed to tasks/done/)
#   1 = task filed to tasks/needs-review/ (or unexpected error)
#   3 = no work attempted / no verdict reached:
#       .agent-stop, lock busy, empty queue, quota threshold, quota cooldown,
#       or a provider rate limit hit mid-task (the task STAYS in tasks/todo/)
#
# Verdict gates (ALL must pass for tasks/done/):
#   0. quota guard                  (preemptive % check if available + reactive cooldown)
#   1. codex exec exit 0
#   2. make check green             (re-run by the runner; the agent's word is not trusted)
#   3. holdout tests green          (optional: HOLDOUT_DIR, tests the agent can never see)
#   4. diff smell check clean       (test-weakening / special-casing tripwires)
#   5. no BLOCKED report, and the diff is non-empty (a silent no-op is not success)
#
# Fix attempts: if verification fails for an ORDINARY reason (codex error, make
# check red, holdout red), the runner gives the agent up to FIX_ATTEMPTS (default
# 0; overnight sets 1) extra rounds with the failure output attached, then re-runs
# every gate. It NEVER retries after a BLOCKED report, a smell-check hit, or a
# provider rate limit — those need a human or a cooldown.
#
# On failure the working tree is ROLLED BACK to the task's base commit, after
# saving the discarded work as logs/<task>-<stamp>-failed.patch. This keeps the
# next task from building on top of a half-finished or possibly-gamed change.
#
# Optional cross-review (advisory, never gates the verdict):
#   CLAUDE_REVIEW=1  pipes the task spec + diff into `claude -p` (read-only) and
#                    saves the verdict to logs/<task>-<stamp>-review.md.
#
# Env overrides:
#   CODEX_BIN / CODEX_ARGS   default: codex / "--sandbox danger-full-access"
#   AGENT_BRANCH             default: agent/auto
#   FIX_ATTEMPTS             extra self-fix rounds after failed verification (default 0)
#   TASK_TIMEOUT             seconds before `codex exec` is interrupted; 0 = no limit (default 0)
#   HOLDOUT_DIR              absolute path OUTSIDE the repo with holdout pytest files
#   CLAUDE_REVIEW / CLAUDE_BIN  enable cross-review / claude binary (default: claude)
#   QUOTA_MAX_PCT            skip new tasks when weekly usage >= this percent (default 90)
#   QUOTA_CMD                command printing weekly USED percent as an integer;
#                            default: scripts/check_quota.sh (best effort, fails open)
#   COOLDOWN_MINUTES         reactive cooldown after a provider-limit error (default 10)
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

# --- runner-owned paths must never be staged --------------------------------
# Without this, `git add -A` at the end would commit the logs, and worse, the
# "silent no-op" gate below could never fire (a fresh log file always makes the
# tree look dirty). Idempotent: appends only what is missing.
touch .gitignore
for p in 'logs/' '.quota-cooldown-until' '.agent-stop' '.venv/'; do
  grep -qxF "${p}" .gitignore 2>/dev/null || echo "${p}" >> .gitignore
done

# --- gate 0: quota guard ----------------------------------------------------
if [[ -f .quota-cooldown-until ]]; then
  NOW="$(date +%s)"
  UNTIL="$(cat .quota-cooldown-until 2>/dev/null || echo 0)"
  if [[ "${UNTIL}" =~ ^[0-9]+$ ]] && (( NOW < UNTIL )); then
    echo "[$(date -Is)] quota cooldown until $(date -d "@${UNTIL}" -Is 2>/dev/null || echo "${UNTIL}") — skipping."
    exit 3
  fi
  rm -f .quota-cooldown-until
fi
QUOTA_MAX_PCT="${QUOTA_MAX_PCT:-90}"
QUOTA_CMD="${QUOTA_CMD:-${REPO_ROOT}/scripts/check_quota.sh}"
if [[ -x "${QUOTA_CMD%% *}" ]] || command -v "${QUOTA_CMD%% *}" >/dev/null 2>&1; then
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
TASK_SLUG="$(echo "${TASK_NAME}" | tr '[:upper:]' '[:lower:]')"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs tasks/done tasks/needs-review
LOG="logs/${TASK_NAME}-${STAMP}.log"
VERIFY_OUT="logs/${TASK_NAME}-${STAMP}-verify.txt"
FAILED_PATCH="logs/${TASK_NAME}-${STAMP}-failed.patch"
BRANCH="${AGENT_BRANCH:-agent/auto}"
echo "[$(date -Is)] task=${TASK_NAME} branch=${BRANCH} log=${LOG}"

# --- get on the agent branch and snapshot the base --------------------------
if git rev-parse --verify "${BRANCH}" >/dev/null 2>&1; then
  git checkout "${BRANCH}" >>"${LOG}" 2>&1
else
  git checkout -b "${BRANCH}" >>"${LOG}" 2>&1
fi
# Commit the .gitignore touch-up (and anything else left over) so that BASE is a
# clean snapshot; otherwise the no-op gate would see permanent noise.
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -m "chore(runner): housekeeping before ${TASK_SLUG}" >>"${LOG}" 2>&1 || true
fi
BASE="$(git rev-parse HEAD)"
echo "base=${BASE}" >>"${LOG}"

# Snapshot the untracked files that already existed BEFORE this task, so that a
# rollback can tell the agent's new files apart from your own uncommitted work.
PRE_UNTRACKED="logs/${TASK_NAME}-${STAMP}-pre-untracked.txt"
git ls-files --others --exclude-standard | sort > "${PRE_UNTRACKED}"

CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_ARGS="${CODEX_ARGS:---sandbox danger-full-access}"
TASK_TIMEOUT="${TASK_TIMEOUT:-0}"
MAX_ATTEMPTS=$(( 1 + ${FIX_ATTEMPTS:-0} ))
ATTEMPT=1
RATELIMITED=0
REASON=""
CODEX_STATUS=0
CHECK_STATUS=0
HOLDOUT_STATUS=0
SMELL_STATUS=0
BLOCKED=0
NOCHANGE=0

while :; do
  echo "--- attempt ${ATTEMPT}/${MAX_ATTEMPTS} ---" >>"${LOG}"

  # --- gate 1: run the agent ------------------------------------------------
  if (( ATTEMPT == 1 )); then
    PROMPT="$(cat "${TASK_FILE}")"
  else
    PROMPT="Your previous attempt at the task below did not pass independent
verification. Failing gates: ${REASON}. Verification output (tail):

$(tail -n 60 "${VERIFY_OUT}" 2>/dev/null)

Diagnose and fix the current working tree so that verification GENUINELY passes,
honoring AGENTS.md — in particular, never weaken tests or verification to get
green. If the failure reveals the task is impossible or self-contradictory,
report BLOCKED in PROGRESS.md instead. Original task spec follows:

$(cat "${TASK_FILE}")"
  fi

  set +e
  if [[ "${TASK_TIMEOUT}" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    timeout --signal=INT --kill-after=60 "${TASK_TIMEOUT}" \
      "${CODEX_BIN}" exec ${CODEX_ARGS} "${PROMPT}" >>"${LOG}" 2>&1
  else
    # shellcheck disable=SC2086
    "${CODEX_BIN}" exec ${CODEX_ARGS} "${PROMPT}" >>"${LOG}" 2>&1
  fi
  CODEX_STATUS=$?
  set -e
  if [[ ${CODEX_STATUS} -eq 124 ]]; then
    echo "codex exec TIMED OUT after ${TASK_TIMEOUT}s" | tee -a "${LOG}"
  fi
  echo "codex exec exit=${CODEX_STATUS}" >>"${LOG}"

  # Provider-limit detection: schedule a cooldown and never burn a fix attempt on it.
  if [[ ${CODEX_STATUS} -ne 0 ]]; then
    if tail -n 40 "${LOG}" | grep -qiE '(usage|rate) ?limit(ed)? *(reached|exceeded|hit)|exceeded your current quota|too many requests|429'; then
      COOLDOWN_MINUTES="${COOLDOWN_MINUTES:-10}"
      date -d "+${COOLDOWN_MINUTES} minutes" +%s > .quota-cooldown-until
      RATELIMITED=1
      echo "provider limit detected — cooldown ${COOLDOWN_MINUTES}min" | tee -a "${LOG}"
    fi
  fi

  # --- gate 2: independent verification -------------------------------------
  set +e
  make check >"${VERIFY_OUT}" 2>&1
  CHECK_STATUS=$?
  set -e
  cat "${VERIFY_OUT}" >>"${LOG}"
  echo "make check exit=${CHECK_STATUS}" >>"${LOG}"

  # --- gate 3: holdout tests the agent has never seen ------------------------
  HOLDOUT_STATUS=0
  if [[ -n "${HOLDOUT_DIR:-}" && -d "${HOLDOUT_DIR:-}" ]]; then
    set +e
    pytest -q "${HOLDOUT_DIR}" >>"${VERIFY_OUT}" 2>&1
    HOLDOUT_STATUS=$?
    set -e
    echo "holdout exit=${HOLDOUT_STATUS}" >>"${LOG}"
  fi

  # --- gate 4: diff smell check ---------------------------------------------
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

  # --- gate 5: BLOCKED report / silent no-op --------------------------------
  BLOCKED=0
  if git diff "${BASE}" -- PROGRESS.md 2>/dev/null | grep -E '^\+' | grep -q 'BLOCKED'; then
    BLOCKED=1
    echo "agent reported BLOCKED in PROGRESS.md" >>"${LOG}"
  fi
  # Runner-owned paths are excluded so that a fresh log file cannot disguise a
  # no-op as real work. (The .gitignore guard above is the primary defence; this
  # is the belt to its braces.)
  NOCHANGE=0
  CHANGED_VS_BASE="$(git diff --name-only "${BASE}" -- . ':(exclude)logs' 2>/dev/null || true)"
  DIRTY_NOW="$(git status --porcelain -- . ':(exclude)logs' 2>/dev/null || true)"
  if [[ -z "${CHANGED_VS_BASE}" && -z "${DIRTY_NOW}" ]]; then
    NOCHANGE=1
    echo "no changes made by the agent" >>"${LOG}"
  fi

  # --- verdict for this attempt ---------------------------------------------
  VERDICT="done"
  REASON=""
  if [[ ${CODEX_STATUS} -ne 0 || ${CHECK_STATUS} -ne 0 || ${HOLDOUT_STATUS} -ne 0 || ${SMELL_STATUS} -ne 0 ]]; then
    VERDICT="needs-review"
    REASON="codex=${CODEX_STATUS} check=${CHECK_STATUS} holdout=${HOLDOUT_STATUS} smell=${SMELL_STATUS}"
  fi
  if [[ ${BLOCKED} -eq 1 ]]; then
    VERDICT="needs-review"; REASON="blocked"
  fi
  if [[ ${NOCHANGE} -eq 1 ]]; then
    VERDICT="needs-review"; REASON="no changes"
  fi

  # Retry policy: only ordinary failures earn a fix attempt.
  if [[ "${VERDICT}" == "done" ]]; then break; fi
  if [[ ${BLOCKED} -eq 1 || ${SMELL_STATUS} -ne 0 || ${RATELIMITED} -eq 1 ]]; then break; fi
  if (( ATTEMPT >= MAX_ATTEMPTS )); then break; fi
  ATTEMPT=$(( ATTEMPT + 1 ))
  echo "[$(date -Is)] verification failed (${REASON}) — giving the agent a fix attempt."
done

# --- rate limit: keep the task in the queue and let the supervisor sleep -----
# Exiting here (before archiving) is what makes a 1am rate limit cost minutes
# instead of the whole night: nothing is filed, nothing is consumed.
if [[ ${RATELIMITED} -eq 1 ]]; then
  if [[ -n "$(git status --porcelain -- . ':(exclude)logs' 2>/dev/null || true)" ]]; then
    git add -A
    git commit -m "chore(wip): partial work on ${TASK_SLUG} before rate limit" >>"${LOG}" 2>&1 || true
  fi
  echo "[$(date -Is)] ${TASK_NAME}: rate limited — left in tasks/todo, cooling down (log: ${LOG})."
  exit 3
fi

# --- distilled reason for humans --------------------------------------------
REASON_SHORT="ok"
if   [[ ${BLOCKED} -eq 1 ]];        then REASON_SHORT="blocked"
elif [[ ${SMELL_STATUS} -ne 0 ]];   then REASON_SHORT="smell check"
elif [[ ${NOCHANGE} -eq 1 ]];       then REASON_SHORT="no changes"
elif [[ ${CHECK_STATUS} -ne 0 ]];   then REASON_SHORT="check failed"
elif [[ ${HOLDOUT_STATUS} -ne 0 ]]; then REASON_SHORT="holdout failed"
elif [[ ${CODEX_STATUS} -ne 0 ]];   then REASON_SHORT="agent error"
fi

# --- optional cross-review (advisory) ---------------------------------------
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

# --- on failure: save the work, then roll the tree back to BASE -------------
if [[ "${VERDICT}" != "done" ]]; then
  # A BLOCKED report is the one piece of a failed attempt worth keeping in git:
  # it is the agent's explanation of why the spec cannot be met, and it is the
  # first thing you will want to read in the morning.
  if [[ ${BLOCKED} -eq 1 && -f PROGRESS.md ]]; then
    cp PROGRESS.md "logs/${TASK_NAME}-${STAMP}-PROGRESS.md"
  fi
  git add -A >>"${LOG}" 2>&1 || true
  git diff --binary --cached "${BASE}" > "${FAILED_PATCH}" 2>/dev/null || true
  git reset --hard "${BASE}" >>"${LOG}" 2>&1 || true
  # Remove ONLY the files that appeared during this task. A blanket `git clean`
  # would also delete untracked work of yours that predates the run.
  git ls-files --others --exclude-standard | sort > "${PRE_UNTRACKED}.post"
  comm -13 "${PRE_UNTRACKED}" "${PRE_UNTRACKED}.post" \
    | grep -v -e '^\.venv/' -e '^logs/' -e '^tasks/' \
    | tr '\n' '\0' | xargs -0 -r rm -f -- 2>>"${LOG}" || true
  mkdir -p logs tasks/todo tasks/done tasks/needs-review
  if [[ ${BLOCKED} -eq 1 && -f "logs/${TASK_NAME}-${STAMP}-PROGRESS.md" ]]; then
    cp "logs/${TASK_NAME}-${STAMP}-PROGRESS.md" PROGRESS.md
    echo "kept the agent's BLOCKED note in PROGRESS.md" >>"${LOG}"
  fi
  echo "rolled back to ${BASE}; discarded work saved to ${FAILED_PATCH}" >>"${LOG}"
fi

# --- file the task & bookkeeping commit -------------------------------------
if [[ "${VERDICT}" == "done" ]]; then
  DEST="tasks/done/$(basename "${TASK_FILE}")"
  MSG="chore(tasks): complete ${TASK_SLUG}"
else
  DEST="tasks/needs-review/$(basename "${TASK_FILE}")"
  MSG="chore(tasks): ${TASK_SLUG} needs review: ${REASON_SHORT}"
fi
git mv "${TASK_FILE}" "${DEST}" 2>>"${LOG}" || mv "${TASK_FILE}" "${DEST}"
git add -A
git commit -m "${MSG}" >>"${LOG}" 2>&1 || true

if [[ "${PUSH:-0}" == "1" ]]; then
  git push -u origin "${BRANCH}" >>"${LOG}" 2>&1 || echo "push failed (no remote?)" >>"${LOG}"
fi

echo "[$(date -Is)] ${TASK_NAME}: ${VERDICT} — ${REASON_SHORT} (details: ${REASON:-all gates green}; log: ${LOG})"
if [[ "${VERDICT}" == "done" ]]; then exit 0; else exit 1; fi