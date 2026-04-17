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
| M05 | FastAPI service + async worker + DB | done |
| M06 | Dockerfile + docker-compose | done |
| M07 | Benchmark harness (RAGTruth / HaluEval) | done |
| M08 | Analysis + figures | done |
| M09 | CLI + Streamlit demo | done |
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

## Command line

```bash
rageval eval --question "What did Aurora carry?" \
             --answer "Aurora carried 4 instruments." \
             --context "Aurora carried 3 instruments." \
             --fail-under 0.8        # exit 1 when faithfulness is below the bar (CI gate)

rageval eval --json request.json     # same thing from a file
rageval batch inputs.jsonl --out results.jsonl
rageval serve                        # start the API in this process
```

`eval` prints a per-claim table (claim, verdict, reason), the scores, the judge model, the
privacy mode, and timings. `batch` writes the benchmark row schema minus the human label, so
its output feeds the same analysis as `bench.run`. A full walkthrough with expected output is
in [docs/demo.md](docs/demo.md).

## Dashboard

```bash
pip install -e ".[demo]"
make demo                            # streamlit run apps/dashboard.py
```

Two tabs: **Evaluate** shows the sanitized preview (masked spans in bold — what the judge
actually receives) before running the claim table and scores, and **Results** renders the
benchmark report inline from a results JSONL. Judge base URL, model, privacy mode, and metrics
are editable in the sidebar. All of its logic lives in `slm_rag_eval.demo`, so the app file is
only input collection and rendering.

## Run the service

`make run` serves the API on :8000 with the worker embedded in the same process. Submitting
returns immediately with a job id; the worker evaluates in the background.

```bash
make run   # in another terminal

curl -s localhost:8000/healthz
# {"status":"ok"}

JOB=$(curl -s -X POST localhost:8000/v1/evaluations \
  -H 'content-type: application/json' \
  -d '{"question":"Which material shields the module?",
       "answer":"The module uses a ceramic shield.",
       "contexts":["The module is protected by a ceramic shield."],
       "options":{"metrics":["faithfulness","relevance"]}}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

curl -s localhost:8000/v1/evaluations/$JOB
# {"job_id":"...","status":"done","result":{"faithfulness":1.0,...},"error":null}
```

`options` is optional; anything omitted falls back to the `SLMEVAL_*` defaults. Status moves
`queued → running → done | error`; a job whose judge call fails twice ends as `error` with the
message. An unknown job id returns 404.

| Environment variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///data/jobs.db` | Job store; point it at Postgres in production |
| `WORKER_EMBEDDED` | `1` | Run the worker inside the API process |
| `WORKER_CONCURRENCY` | `2` | Jobs evaluated in parallel per worker |

Set `WORKER_EMBEDDED=0` and run the worker as its own process when you want to scale it
separately:

```bash
python -m slm_rag_eval.service.worker
```

**Privacy at rest:** with `SLMEVAL_PRIVACY_MODE=mask` the request is sanitized at the API
boundary, *before* the job row is written — so the database, the judge prompts, and the stored
result all hold placeholders rather than PII. The placeholder mapping is never persisted, which
means results served by the API stay masked; use the in-process pipeline (or the CLI) when you
need de-anonymized display text.

## Run with Docker

From a clean checkout to a scored answer, with the judge, the database, and the worker all
in containers:

```bash
cp .env.example .env          # edit SLMEVAL_MODEL / JUDGE_MODEL if you want another judge
make docker-build             # multi-stage build; the spaCy model is baked into the image
make docker-up                # api, worker, postgres, ollama, and a one-shot model pull
make docker-logs              # follow api + worker; ollama-init exits once the pull is done

# the first `ollama pull` downloads several GB — wait for it before submitting
curl -s localhost:8000/healthz

JOB=$(curl -s -X POST localhost:8000/v1/evaluations \
  -H 'content-type: application/json' \
  -d '{"question":"Which material shields the module?",
       "answer":"The module uses a ceramic shield.",
       "contexts":["The module is protected by a ceramic shield."]}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

curl -s localhost:8000/v1/evaluations/$JOB
make docker-down              # stop everything; named volumes keep the model and the data
```

The `api` service sets `WORKER_EMBEDDED=0` and the `worker` service runs
`python -m slm_rag_eval.service.worker`, so they scale independently. `db` is `postgres:16`
behind a `pg_isready` healthcheck and both app services wait for it. `ollama-init` pulls
`JUDGE_MODEL` once and exits. To give the judge a GPU, uncomment the `deploy.resources`
block on the `ollama` service (needs the NVIDIA container toolkit on the host).

Unit tests never need Docker: `make check` runs entirely in-process.

## Benchmarks

The harness scores a human-labeled dataset with one judge and writes JSONL rows that M08
turns into a report.

```bash
python scripts/download_ragtruth.py     # -> data/ragtruth/ (gitignored)
python scripts/download_halueval.py     # -> data/halueval/

python -m slm_rag_eval.bench.run --dataset ragtruth --judge slm --limit 200 --out results/
python -m slm_rag_eval.bench.run --help
```

Each run writes `results/<dataset>_<judge>.jsonl` (one row per sample) and
`results/<dataset>_<judge>.manifest.json` (parameters, git sha, UTC timestamp, failure
count). Runs are **resumable**: rerunning the same dataset/judge pair skips sample ids
already present in the rows file, so an interrupted 200-sample run continues where it
stopped. A sample that raises is logged, counted in the manifest, and skipped — one bad
sample never ends a run. `--privacy-mode` overrides the configured masking mode for the run.

The loaders never touch the network: they read the cached files and raise a clear error
naming the download script when the cache is missing.

### SLM vs cloud comparison

```bash
# local judge — the one that keeps data in-house
python -m slm_rag_eval.bench.run --dataset ragtruth --judge slm --limit 200 --out results/

# cloud baseline for the same samples
export CLOUD_BASE_URL=https://api.example.com/v1
export CLOUD_MODEL=<model>
export CLOUD_API_KEY=<key>
python -m slm_rag_eval.bench.run --dataset ragtruth --judge cloud --limit 200 --out results/
```

> ⚠️ **The cloud judge is for PUBLIC benchmark data only.** Never point it at private or
> production data. Keeping sensitive retrieved contexts away from third-party APIs is the
> entire point of this project; the cloud judge exists only to produce a comparison baseline
> on already-public datasets.

### Analysis and figures

```bash
python -m slm_rag_eval.bench.analyze results/*.jsonl --out report/ \
  --cloud-input-cost-per-1m 5 --cloud-output-cost-per-1m 15
```

Writes `report/summary.md` plus three PNGs (ROC curves with one line per judge, faithfulness
distributions, latency box plot). The report contains:

- **Detection quality** per judge. A sample is predicted hallucinated when
  `faithfulness < t`; `t` is swept over `[0, 1]` in steps of `0.05` with precision, recall,
  F1, and balanced accuracy at every step, plus the threshold-independent ROC-AUC. The
  best-F1 threshold is reported per judge (ties go to the lower threshold). Rows the judge
  could not score are counted as **unscored** and excluded from the threshold metrics rather
  than being silently treated as zeros.
- **Agreement** between judges over the samples both scored: Pearson and Spearman on the
  faithfulness scores, and Cohen's kappa on the binary calls each judge makes at *its own*
  best threshold.
- **Efficiency**: median and p95 latency per sample, mean tokens, and an estimated cost table.
  The cloud price is supplied with the two CLI flags; the local judge is reported at zero
  marginal cost, with the caveat that it occupies hardware you already pay for.

A sample report generated from the synthetic fixtures lives in
[`docs/sample_report/`](docs/sample_report/summary.md) — it illustrates the output shape, and
its numbers mean nothing beyond that.

### Datasets, licenses, citations

- **RAGTruth** (primary) — <https://github.com/ParticleMedia/RAGTruth>, MIT License,
  © 2023 Particle Media. Response-level label: a response counts as hallucinated when it
  carries at least one annotated span. Niu et al., *RAGTruth: A Hallucination Corpus for
  Developing Trustworthy Retrieval-Augmented Language Models*, ACL 2024.
- **HaluEval** (secondary, QA subset) — <https://github.com/RUCAIBox/HaluEval>, MIT License,
  © 2020 RUCAIBox. Each source row yields two samples, the correct answer (not hallucinated)
  and the hallucinated answer. Li et al., *HaluEval: A Large-Scale Hallucination Evaluation
  Benchmark for Large Language Models*, EMNLP 2023.

Downloaded data stays in `data/` and is gitignored; nothing from either dataset is committed
to this repository, and the test fixtures under `tests/data/` are entirely synthetic.

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
