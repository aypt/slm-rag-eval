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
| M04 | Privacy layer (Presidio) | done |
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
source .venv/bin/activate   # same interpreter for setup and check — see the note below
make setup   # install package + dev tools
make check   # ruff + mypy + pytest — the definition of "green"
make run     # dev server on :8000 (GET /healthz)
```

Run `make setup` and `make check` under the **same** interpreter. The task runner puts
`.venv/bin` on `PATH` before gating on `make check`, so a `make setup` that lands anywhere
else (a login shell that dropped the venv, `~/.local`, a system Python) installs
dependencies the gate cannot see — which is exactly how M04 first failed.

Unit tests never touch the network: all judge calls go through the `FakeLLMClient`
fixture in `tests/conftest.py`.

`make setup` also downloads spaCy's `en_core_web_lg` English model. Presidio uses this
model for named-entity recognition, including people and locations; installing the Python
packages alone is not enough to run the default analyzer.

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

## Privacy layer

The metric registry treats the judge boundary as untrusted. With the default
`privacy_mode="mask"`, it runs the question, answer, and every retrieved context through
Microsoft Presidio before making any judge call. Each distinct detected surface form gets a
typed placeholder such as `<PERSON_1>` or `<EMAIL_ADDRESS_1>`, and repeated values use the
same placeholder across the whole request. The placeholder-to-original mapping remains only
in memory and is used after all judge calls to restore verdict text for trusted local display;
it must not be logged or persisted.

This protects configured PII that Presidio detects from being included in outbound judge
prompts. It does not make arbitrary text anonymous: statistical NER can miss entities,
unsupported entity types are not masked, and indirect identifiers or sensitive facts that do
not match a recognizer can remain. Review detector coverage for the deployment's language and
data, keep sensitive persistence and logs inside the trusted boundary, and use an in-process
judge when missed PII cannot be tolerated. Setting privacy mode to `off` deliberately sends
the original request to the judge.

The defaults detect `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IP_ADDRESS`,
`LOCATION`, and `US_SSN` at a minimum confidence score of `0.4`. Configure them with
`SLMEVAL_PRIVACY_ENTITIES` (a JSON list of Presidio entity names) and
`SLMEVAL_PRIVACY_SCORE_THRESHOLD` (from `0.0` to `1.0`). A lower threshold favors recall and
may mask more non-PII text; a higher threshold favors precision and may miss more PII.

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
| `SLMEVAL_PRIVACY_ENTITIES` | Presidio PII list above | Entity names to detect (JSON list) |
| `SLMEVAL_PRIVACY_SCORE_THRESHOLD` | `0.4` | Minimum detector confidence (`0.0`–`1.0`) |

Keep `SLMEVAL_PRIVACY_MODE=mask` when the configured judge is outside the trusted process.
