# AGENTS.md — slm-rag-eval

## What this project is
A privacy-preserving pipeline that evaluates RAG outputs for faithfulness/hallucination
using small language models (SLMs, 3–9B) as judges, instead of sending sensitive
retrieved contexts to cloud LLMs. MEng project; deliverables: working pipeline, Docker
Compose deployment, benchmark report comparing SLM judges against a cloud-judge baseline.

**This project IS an evaluation system.** If evaluation logic is ever fitted to specific
test or benchmark examples, every downstream number (RAGTruth comparison, the final
report, the thesis conclusions) is poisoned. Gaming a metric here is not a code-quality
issue; it is an academic-integrity issue.

## Honest failure is a success
You are explicitly authorized to stop and report. If a task is impossible,
self-contradictory, or underspecified, write your finding and exit WITHOUT changing
code. A reported blocker is a SUCCESSFUL outcome. A green check obtained by weakening
verification is a FAILED outcome, and worse than doing nothing.

To report a blocker: append a section `## <TASK-ID> — BLOCKED` to PROGRESS.md
explaining the conflict, change nothing else, and finish. The runner will route the
task to human review — that is the intended path, not a punishment.

## Layout
- `src/slm_rag_eval/core/` — Pydantic schemas and Settings.
- `src/slm_rag_eval/llm/` — backend-agnostic LLM client (OpenAI-compatible HTTP).
- `src/slm_rag_eval/metrics/` — faithfulness, relevance, registry.
- `src/slm_rag_eval/privacy/` — Presidio-based PII sanitizer.
- `src/slm_rag_eval/service/` — FastAPI app, DB, async worker.
- `src/slm_rag_eval/bench/` — dataset loaders, benchmark runner, analysis.
- `tests/` mirrors src. `tasks/` holds work orders. `docs/plan.md` is the master plan.

## Commands
- `make setup` — install package + dev tools (editable, `[dev]` extras).
- `make check` — ruff + mypy + pytest. **MUST be green before work counts as done.**
- `make test` — pytest only.
- `make run` — dev server on :8000.

## Never weaken verification
Your work is judged by gates you partly cannot see (a holdout test suite and a diff
audit run outside this workspace). Passing the visible checks by trickery will still
fail the run. Therefore:

1. Do not modify, delete, skip, or xfail any existing test. You are expected to ADD
   tests; never remove or edit existing assertions. Do not change existing fixtures in
   `tests/conftest.py`; put new helpers in new files under `tests/`.
2. Do not touch pytest/ruff/mypy configuration (`[tool.*]` in pyproject.toml), the CI
   workflow's check steps, or the Makefile `check`/`lint`/`type`/`test` targets. You
   may extend other Makefile targets (e.g. `setup`) only when the task explicitly
   says so.
3. No new `# type: ignore` or `# noqa` in `src/`. In tests they are allowed only on
   intentionally-invalid-input lines, with a comment saying why.
4. Never special-case values, inputs, or types that appear in tests or benchmark
   data. Implement the general behaviour; passing tests must be a CONSEQUENCE of
   correctness, never the target.
5. Do not override `__eq__`, `__hash__`, `__bool__`, or other dunder methods to
   satisfy an assertion.
6. Never fit evaluation logic (claim extraction, verification, scoring, thresholds)
   to specific benchmark examples.

## Other hard rules
1. **Unit tests never touch the network or a real model.** All judge calls in tests go
   through the `FakeLLMClient` fixture (`tests/conftest.py`), which records every
   prompt it receives.
2. **Privacy invariant:** when `privacy_mode="mask"`, no raw PII string may appear in
   any outbound prompt or in persisted request data. Tests assert this — never weaken
   or delete them.
3. Test fixtures use only synthetic/fake personal data. Never paste real names,
   emails, phone numbers, or dataset text containing PII into fixtures.
4. All judge I/O is structured: request JSON via `generate_json()` with a Pydantic
   schema and a bounded repair-retry loop. Do not parse free text with regex.
5. Python 3.11+, type hints on public functions, ruff clean, line length 100.
6. **Never edit files under `tasks/` or `scripts/`.** If a spec there seems wrong,
   report it in PROGRESS.md and stop.
7. If a spec is ambiguous (but not contradictory), choose the simplest reasonable
   interpretation, record the decision in PROGRESS.md, and continue.
8. Never commit secrets. `.env` is gitignored; use `.env.example` for documentation.

## Git hygiene
- Small commits with imperative messages ("Add faithfulness batch verification").
- **Never `git add -A` or `git add .`** — stage only the files you intentionally
  changed, by name.
- Keep `.gitignore` complete; never commit caches, artifacts, virtualenvs, or data.
- Work only on the branch you were started on; **never commit to main**.

## Task system
Work orders live in `tasks/todo/` (one Markdown file each; lexicographic order =
dependency order; `tasks/TEMPLATE.md` shows the format). The runner
`scripts/run_next_task.sh` feeds the next file to the agent, then independently
verifies: `make check`, a holdout test suite outside this workspace, and a diff audit
for the tricks listed above. All gates green → `tasks/done/`; anything else →
`tasks/needs-review/` for the human's morning review.

## When you finish a task
1. Run `make check` until green.
2. Append a section to PROGRESS.md: the task's acceptance criteria as a checklist,
   each item with concrete evidence (test names or command output).
3. End with a short summary: what changed, how it was verified, open TODOs.
