# slm-rag-eval

Privacy-preserving evaluation of RAG (retrieval-augmented generation) outputs for
**faithfulness / hallucination**, using **small language models (SLMs, 3–9B)** as judges —
so sensitive retrieved contexts never have to leave your infrastructure.

MEng project (ECE, Western University). Deliverables: working pipeline, Docker Compose
deployment, SLM-vs-cloud-judge benchmark report.

## Status

Built task-by-task by a coding agent from the specs in `tasks/` (see `docs/plan.md` for
the full plan and `docs/automation.md` for how the autonomous runner works).

| Task | Scope | State |
|---|---|---|
| M00 | Scaffold, CI, test harness | done |
| M01 | LLM client abstraction (Ollama/vLLM/OpenAI-compatible) | done |
| M02 | Faithfulness metric (claims + verification) | todo |
| M03 | Relevance metric + registry | todo |
| M04 | Privacy layer (Presidio) | todo |
| M05 | FastAPI service + async worker + DB | todo |
| M06 | Dockerfile + docker-compose | todo |
| M07 | Benchmark harness (RAGTruth / HaluEval) | todo |
| M08 | Analysis + figures | todo |
| M09 | CLI + Streamlit demo | todo |
| M10 | Docs, hardening, reproducibility | todo |

## Deploy

First time here? Follow **docs/SETUP.md** step by step (从零部署手册).

## Development

```bash
make setup   # install package + dev tools
make check   # ruff + mypy + pytest — the definition of "green"
make run     # dev server on :8000 (GET /healthz)
```

Unit tests never touch the network: all judge calls go through the `FakeLLMClient`
fixture in `tests/conftest.py`.

## Configuring model backends

Model backends are configured with `SLMEVAL_*` environment variables. For a local Ollama
server exposing its OpenAI-compatible API:

```bash
export SLMEVAL_BASE_URL=http://localhost:11434/v1
export SLMEVAL_MODEL=qwen2.5:7b-instruct
export SLMEVAL_TIMEOUT_S=120
export SLMEVAL_MAX_RETRIES=3
```

`SLMEVAL_API_KEY` is optional and should be left unset for an unsecured local Ollama
server. Set it when the selected OpenAI-compatible backend requires bearer authentication.
