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
| M02 | Faithfulness metric (claims + verification) | done |
| M03 | Relevance metric + registry | done |
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

## Metrics

The metric registry runs faithfulness by default. Pass `metrics=["faithfulness",
"relevance"]` to `metrics.registry.evaluate` to run both; its result retains the
faithfulness verdicts and reports per-metric total latency and model/token metadata.

- **Faithfulness** extracts atomic factual claims from the answer as structured JSON,
  then verifies batches of up to five claims against only the supplied contexts. In
  strict mode, unsupported and uncertain claims both reduce the score. Non-strict mode
  excludes uncertain claims from the denominator. An empty answer, no claims after the
  bounded extraction retry, or an all-uncertain non-strict verdict produces `None`.
- **Relevance** first generates exactly three questions directly answered by the answer,
  then makes one structured rating request comparing each generated question with the
  original on a 0–2 scale. The score is the mean rating divided by two. An empty answer
  produces `None` without calling the judge.

Both metrics use schema-validated `generate_json` calls. Invalid JSON or schema violations
are repaired at most twice and then raise `JSONGenerationError`; backend failures propagate.
Faithfulness also re-asks once when a verification response has the wrong number of verdicts,
then raises `ValueError` if it remains misaligned.

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

Metric and privacy defaults are:

| Environment variable | Default | Meaning |
|---|---|---|
| `SLMEVAL_ENABLED_METRICS` | `["faithfulness"]` | Registry metrics (JSON list) |
| `SLMEVAL_K` | `1` | Faithfulness verification runs used for majority voting |
| `SLMEVAL_STRICT` | `true` | Count uncertain faithfulness verdicts as unsupported |
| `SLMEVAL_PRIVACY_MODE` | `mask` | `mask` or `off` |

The PII sanitizer is delivered in M04. Until then, `mask` and `off` are both accepted and
behave identically; no masking is performed yet.
