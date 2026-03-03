# PROGRESS

Append-only evidence log. Every completed task adds a section here: each acceptance
criterion checked off with concrete evidence (test names, command output).

## M00 · Scaffold — DONE
- [x] `make check` green (ruff + mypy + pytest) — verified locally at scaffold time.
- [x] Package imports; `/healthz` returns 200 (tests/service/test_api.py::test_healthz).
- [x] FakeLLMClient fixture records calls and enforces scripted responses
      (tests/test_fake_llm.py, 2 tests).
- [x] CI workflow present (.github/workflows/ci.yml) running `make check`.
