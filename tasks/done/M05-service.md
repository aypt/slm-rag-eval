# M05 · API service + async worker + persistence

Context: repo slm-rag-eval; M01–M04 done. This file is the complete spec; also
follow AGENTS.md.

Task: turn the pipeline into an async web service in src/slm_rag_eval/service/.

Requirements:
1. FastAPI app (extend service/api.py):
   - POST /v1/evaluations — body: EvalRequest + options {metrics, k, strict,
     privacy_mode}; creates a Job row; returns 202 {job_id, status: "queued"}.
   - GET /v1/evaluations/{job_id} — {job_id, status, result?, error?}; 404 unknown.
   - GET /healthz stays as-is.
2. Persistence (service/db.py): SQLAlchemy 2 async; model Job{id (uuid str), status:
   queued|running|done|error, request_json, result_json, error, created_at,
   updated_at}. Default sqlite+aiosqlite:///data/jobs.db (create data/ if missing);
   DATABASE_URL env overrides (Postgres-ready). Create tables on startup.
3. Worker (service/worker.py, replace stub): an asyncio loop that claims queued
   jobs with an optimistic status update (queued -> running) safe for single-writer
   SQLite and for Postgres, runs registry.evaluate with a judge built from Settings,
   stores results; concurrency via semaphore (WORKER_CONCURRENCY, default 2); a
   failed job is retried once, then marked error with the message. Runnable two
   ways: embedded in the API process via lifespan when WORKER_EMBEDDED=1 (default),
   or standalone via `python -m slm_rag_eval.service.worker`.
4. Privacy at rest: when privacy_mode="mask", persist the sanitized request, never
   the raw one. Add a test asserting the stored request_json contains placeholders
   and not the raw fixture PII.

Testing: httpx.AsyncClient with ASGITransport; judge dependency overridden to
FakeLLMClient; end-to-end test: submit -> embedded worker processes -> poll result
until done (bounded wait); unknown-job 404; error path (judge raising) marks the
job error after one retry.

Definition of done: `make check` green; `make run` serves the API locally and a
curl quickstart in README works; PROGRESS.md updated.
