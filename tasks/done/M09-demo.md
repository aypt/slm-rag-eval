# M09 · Demonstration interface (CLI + Streamlit dashboard)

Context: repo slm-rag-eval; M01–M08 done. This file is the complete spec; also
follow AGENTS.md.

Task:
1. Extend src/slm_rag_eval/cli.py (console script `rageval`):
   - `rageval eval` — flags --question/--answer/--context (repeatable) or --json
     FILE; pretty terminal output: per-claim table (claim, verdict, reason) and
     scores; exit code 1 if faithfulness < --fail-under (default 0.0, i.e. never
     fails unless set).
   - `rageval batch FILE.jsonl --out results.jsonl` — thin wrapper over the
     pipeline with the same row schema as bench.run minus labels.
   - `rageval serve` — starts uvicorn programmatically.
2. Streamlit app apps/dashboard.py in a new optional-dependency group [demo]
   (`make demo` runs it):
   - Tab "Evaluate": paste question/answer/contexts; show the sanitized preview
     with PII placeholders highlighted; claim table with verdicts and reasons;
     score metrics; latency. Judge settings (base_url/model) editable in the
     sidebar.
   - Tab "Results": load a bench results JSONL (file uploader or path input) and
     render the M08 summary tables and figures inline.
   Keep it single-file; factor ALL logic into pure functions imported from the
   package so they are unit-testable; no custom CSS.
3. Docs: write docs/demo.md — a copy-paste walkthrough (commands + expected
   output shape) that a human can follow to record the project demo video; include
   a captured `rageval eval` transcript produced via the test runner.

Testing: CLI via typer.testing.CliRunner with FakeLLMClient injected through an
env/DI seam; dashboard pure functions unit-tested; do not attempt to test
Streamlit rendering itself.

Definition of done: `make check` green; docs/demo.md exists with a real captured
transcript; PROGRESS.md updated.
