# M10 · Documentation, hardening, reproducibility

Context: repo slm-rag-eval; all features merged. This file is the complete spec;
also follow AGENTS.md.

Task:
1. README overhaul: problem statement (privacy-preserving RAG evaluation with SLM
   judges), ASCII architecture diagram, quickstart (local + Docker), configuration
   reference table generated from the Settings model (write a small script that
   emits it and paste the output), metrics explanation, benchmark reproduction
   steps, limitations & future work.
2. Docstrings on all public functions; tighten mypy (disallow_untyped_defs) for
   core/, llm/, metrics/ and fix fallout.
3. Error-path review: consistent exception -> HTTP error mapping in the API;
   structured JSON logging with a request/job id field.
4. `make reproduce`: runs a 20-sample bench + analyze end-to-end against Ollama if
   SLMEVAL_BASE_URL is reachable, otherwise against FakeLLMClient, writing to
   report/repro/. This is the command used in the project demo.
5. Repo hygiene: pin dependency versions (constraints file or exact pins); verify
   the path fresh-clone -> `make setup` -> `make check` -> `make reproduce` by
   actually executing it in a clean temp directory and fixing anything that breaks;
   document the verified transcript in PROGRESS.md.

Definition of done: `make check` green; the fresh-clone walkthrough is documented
verbatim in README and PROGRESS.md; PROGRESS.md updated.
