# M__ · <task title>

Context: repo slm-rag-eval; <preceding tasks> done. This file is the complete spec;
also follow AGENTS.md.

Task: <one paragraph describing the BEHAVIOR to build — describe behaviour, never
"make the checks pass"; green checks must be a consequence, not the target>

Requirements:
1. <numbered, concrete, testable>
2. ...

Testing: <which NEW tests to add; FakeLLMClient only, no network; extend existing
test files only by appending — never edit or remove existing assertions>

Out of scope: <explicitly excluded work, so the diff stays reviewable>

If any requirement conflicts with existing tests or with another requirement, STOP
and report it in PROGRESS.md (`## M__ — BLOCKED`) without changing code. That is a
successful outcome.

Definition of done: `make check` green as a consequence of the implementation;
acceptance checklist appended to PROGRESS.md with evidence (test names / command
output).
