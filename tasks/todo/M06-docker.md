# M06 · Containerization (Dockerfile + docker-compose)

Context: repo slm-rag-eval; M01–M05 done. This file is the complete spec; also
follow AGENTS.md.

Task: containerize the full stack.

Requirements:
1. Multi-stage Dockerfile on python:3.11-slim — builder installs deps and the spaCy
   model; runtime image is non-root; HEALTHCHECK curls /healthz (install curl or use
   python -c urllib fallback).
2. docker-compose.yml services:
   - api: uvicorn on :8000, env from .env, WORKER_EMBEDDED=0, depends_on db and
     ollama (healthy where healthchecks exist).
   - worker: same image, command `python -m slm_rag_eval.service.worker`.
   - db: postgres:16 with named volume and pg_isready healthcheck.
   - ollama: ollama/ollama with a named volume for models; include a commented
     GPU `deploy.resources.reservations.devices` block with a one-line note.
   - ollama-init: one-shot service that pulls the judge model configured in .env
     (JUDGE_MODEL) via `ollama pull`, then exits.
3. .env.example documenting every setting: SLMEVAL_BASE_URL=http://ollama:11434/v1,
   SLMEVAL_MODEL / JUDGE_MODEL, DATABASE_URL pointing at the compose Postgres,
   WORKER_CONCURRENCY, SLMEVAL_PRIVACY_MODE, metric defaults.
4. Make targets: docker-build, docker-up, docker-down, docker-logs.
5. CI: add a job that builds the Docker image (build only; do not run compose).
Constraint: unit tests must still run without Docker.

Definition of done: `docker compose config` validates (add a unit-adjacent check or
document manual verification in PROGRESS.md if the docker CLI is unavailable in the
dev environment); the image builds in CI; README "Run with Docker" section has
copy-paste commands from clean checkout to a successful curl against the api
container; `make check` green; PROGRESS.md updated.
