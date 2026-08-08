# slm-rag-eval

Privacy-preserving evaluation of RAG (retrieval-augmented generation) outputs for
**faithfulness / hallucination**, using **small language models (SLMs, 3–9B)** as judges —
so sensitive retrieved contexts never have to leave your infrastructure.

MEng project (ECE, Western University). Deliverables: working pipeline, Docker Compose
deployment, SLM-vs-cloud-judge benchmark report.

## Status

**Complete.** All eleven build tasks (M00–M10) are in `tasks/done/`, and the final benchmark
ran on rented GPU hardware on 2026-08-08. The numbers below are that run.

## Results

150 stratified RAGTruth test samples, three local judges at Q4_K_M on one
RTX 4090, against two cloud judges. Thresholds are selected on one half of the labelled rows
and measured on the other, so these are **held-out** figures rather than best-case fits.

| Judge | Where | Held-out F1 | κ vs human | Scored | Median latency | Peak VRAM |
|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | cloud | **0.590** | 0.352 | 149/150 | 6.92 s | — |
| Qwen3 8B | local | 0.516 | 0.190 | 131/150 | 26.19 s | 21.98 GiB |
| GPT-5 mini | cloud | 0.514 | 0.214 | 149/150 | 52.57 s | — |
| Qwen3 4B Instruct | local | 0.493 | 0.148 | 145/150 | **5.17 s** | 17.62 GiB |
| Gemma3 4B | local | 0.474 | 0.181 | 106/150 | 7.12 s | **7.49 GiB** |

**What this says.** The best local judge lands within 0.002 F1 of GPT-5 mini on this sample,
and Qwen3 4B Instruct is the most practical operating point — lowest latency, 96.7% coverage,
and the smallest cost per scored evaluation. But no judge, cloud or local, exceeded **κ 0.352**
against human labels. Every judge also ran high-recall and low-precision, which suits screening
and not autonomous gating. **Nothing here supports unattended use for high-stakes decisions.**

**Coverage is part of the result.** Rows a judge could not score are reported, never silently
dropped, and the *Scored* column above is the denominator for that judge's metrics. Gemma3 4B
left 44 of 150 unscored — 42 because it would not echo claims verbatim, so verdicts could not be
aligned to claims, and 2 from truncation. Qwen3 8B left 19 unscored, 16 of which raised an error
(14 hit the 16,384-token completion budget, 2 stayed misaligned). A misattributed verdict is a
wrong evaluation that looks correct, so the alignment check stays strict and the loss is
reported instead of being repaired away.

**Privacy.** The Presidio layer detected PII at precision 0.988 / recall 0.944 on a 73-document
synthetic corpus. Masking is not free: it cost Qwen3 8B **0.119 held-out F1**. That trade is
measured rather than assumed, and it is the number to budget for.

Full artifacts — raw rows, manifests, figures, environment, and a SHA-256 manifest — are
produced by one command (see [Reproducing the results](#reproducing-the-results)). They are not
committed here; `/report*/` is gitignored.

## How it works

A retrieved-context evaluation never leaves your infrastructure: the request is masked at the
trust boundary, and the judge is a small model you host.

```text
  client ──POST /v1/evaluations──▶ FastAPI ──▶ Presidio sanitizer ──▶ job row (masked)
                                     │                                    │
                                     │ 202 {job_id}                       │ claim (UPDATE)
                                     ▼                                    ▼
  client ──GET /v1/evaluations/{id}──▶ job row ◀──── result ────── async worker
                                                                          │
                                             ┌────────────────────────────┴────────────┐
                                             ▼                                         ▼
                                    faithfulness metric                        relevance metric
                                    claims → verify(≤5)                        3 questions → rate
                                             └────────────────┬────────────────────────┘
                                                              ▼
                                                    SLM judge (Ollama /
                                                    any OpenAI-compatible)
                                                              │
   bench.run ──▶ rows.jsonl ──▶ bench.analyze ──▶ report/summary.md + figures
```

The same pipeline is reachable three ways: `rageval` on the command line, the FastAPI service,
and the Streamlit dashboard. `bench.run` drives it over labeled datasets to compare an SLM
judge against a cloud baseline.

## Deploy

Quickest path to a running stack is [Run with Docker](#run-with-docker). To work on the code
instead, start with [Development](#development). For the full benchmark on a rented GPU host,
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) is the operator's procedure, including the preflight gates
and the cost controls.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate   # one interpreter for everything
make setup                  # package + dev/analysis extras + the spaCy model
make check                  # ruff + mypy + pytest — the definition of "green"
make run                    # dev server on :8000 (GET /healthz)
make reproduce              # 20-sample bench + analysis into report/repro/

pip install -e ".[dev,analysis]" -c constraints.txt   # exact pinned versions
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

Writes `report/summary.md` plus five PNGs: held-out F1 with coverage per judge, ROC curves,
faithfulness distributions, a latency box plot, and a quality-against-latency-and-cost
trade-off scatter. The report contains:

- **Detection quality** per judge, reported twice. A sample is predicted hallucinated when
  `faithfulness < t`; `t` is swept over `[0, 1]` in steps of `0.05` with precision, recall,
  F1, and balanced accuracy at every step, plus the threshold-independent ROC-AUC. The
  in-sample table picks the best-F1 threshold on the rows it reports (ties go to the lower
  threshold), which makes it an upper bound. The **held-out** table picks the threshold on
  half the labelled rows and measures on the other half, split by a hash of the sample id so
  every judge is measured on the same held-out samples — those are the numbers to quote. Rows
  the judge could not score are counted as **unscored** and excluded from the threshold
  metrics rather than being silently treated as zeros.
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
export SLMEVAL_MODEL=qwen3:4b-instruct   # see Results for the three judges benchmarked
export SLMEVAL_TIMEOUT_S=120
export SLMEVAL_MAX_RETRIES=3
export SLMEVAL_MAX_TOKENS=16384          # long RAGTruth contexts truncate below this
export SLMEVAL_DISABLE_THINKING=true     # hybrid-reasoning judges answer directly
```

The built-in default is still `qwen2.5:7b-instruct`; the judges actually measured were
`qwen3:8b`, `qwen3:4b-instruct`, and `gemma3:4b`, all Q4_K_M. Set `SLMEVAL_MODEL` explicitly
rather than relying on the default, and record the tag and digest with any number you report —
two artifacts of the same base model do not necessarily produce the same judgments.

`SLMEVAL_API_KEY` is optional and should be left unset for an unsecured local Ollama
server. Set it when the selected OpenAI-compatible backend requires bearer authentication.

Keep `SLMEVAL_PRIVACY_MODE=mask` when the configured judge is outside the trusted process.

### Configuration reference

Every setting, emitted from the `Settings` model by
`python scripts/emit_config_table.py` — regenerate and paste after changing the model, and
`tests/deploy/` will fail if `.env.example` stops covering it.

<!-- generated: python scripts/emit_config_table.py -->

| Environment variable | Default | Description |
|---|---|---|
| `SLMEVAL_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible judge endpoint. |
| `SLMEVAL_API_KEY` | (unset) | Bearer token; leave unset for a local Ollama. |
| `SLMEVAL_MODEL` | `qwen2.5:7b-instruct` | Judge model name. |
| `SLMEVAL_TIMEOUT_S` | `120.0` | Per-request timeout in seconds. |
| `SLMEVAL_MAX_RETRIES` | `3` | Attempts for transient transport failures. |
| `SLMEVAL_MAX_TOKENS` | `2048` | Completion token budget per judge call; raise it for long contexts. |
| `SLMEVAL_DISABLE_THINKING` | `true` | Append a no-thinking directive so hybrid-reasoning judges answer directly. |
| `SLMEVAL_ENABLED_METRICS` | `["faithfulness"]` | Metrics the registry runs by default (JSON list). |
| `SLMEVAL_K` | `1` | Self-consistency verification runs. |
| `SLMEVAL_STRICT` | `true` | Count uncertain verdicts as unsupported. |
| `SLMEVAL_PRIVACY_MODE` | `mask` | `mask` sanitizes every request before it leaves the process. |
| `SLMEVAL_PRIVACY_ENTITIES` | `["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IP_ADDRESS", "LOCATION", "US_SSN"]` | Presidio entity types to detect (JSON list). |
| `SLMEVAL_PRIVACY_SCORE_THRESHOLD` | `0.4` | Minimum detector confidence. |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/jobs.db` | Job store; SQLite by default, Postgres in the compose deployment. |
| `WORKER_EMBEDDED` | `true` | Run the worker inside the API process. |
| `WORKER_CONCURRENCY` | `2` | Jobs evaluated in parallel per worker. |

`JUDGE_MODEL` is compose-only: the `ollama-init` service pulls it. Keep it equal to
`SLMEVAL_MODEL`.

## Reproducing the results

```bash
make reproduce          # 20-sample bench + analysis into report/repro/
```

It uses the configured Ollama judge when `SLMEVAL_BASE_URL` answers, and an offline stub
otherwise, so it runs on any machine. **The stub is not an evaluator** — it marks every claim
`uncertain`, which scores 0.0 across the board in strict mode. That is deliberate: a smoke
test must be impossible to mistake for a result. For real numbers, start a judge first.

Full benchmark reproduction:

```bash
python scripts/download_ragtruth.py
python -m slm_rag_eval.bench.run --dataset ragtruth --judge slm   --limit 200 --out results/
python -m slm_rag_eval.bench.run --dataset ragtruth --judge cloud --limit 200 --out results/
python -m slm_rag_eval.bench.analyze results/*.jsonl --out report/
```

Each run records the git sha, UTC timestamp, and parameters in its manifest. Dependencies are
pinned in `constraints.txt`; install with `-c constraints.txt` to reproduce the exact
environment the numbers came from.

### The full experiment, in one command

`bench.campaign` runs everything the report needs and gates each expensive step behind a cheap
one, so a misconfiguration fails in seconds rather than after hours of GPU time:

```bash
export SLMEVAL_BASE_URL=http://localhost:11434/v1
export SLMEVAL_MAX_TOKENS=16384      # 8192 truncated Qwen3 8B on real samples
export SLMEVAL_PRIVACY_MODE=mask
export SLMEVAL_DISABLE_THINKING=true

python -m slm_rag_eval.bench.campaign \
  --model qwen3:8b --model qwen3:4b-instruct --model gemma3:4b \
  --include-rows report/cloud-baseline/<provider>/ragtruth_cloud.jsonl \
  --limit 150 --ablation-limit 100 --seed 20260806 \
  --out report/experiment
```

In order: `make check`, dataset-lock verification, model-artifact freeze, a per-model preflight
(one real sample, catching truncation and misconfiguration before the expensive leg), the
primary matrix, the masking ablation, PII detection, the Compose check, and a checksummed
bundle. Blocking steps stop the run; the rest degrade and are recorded. Budget 4–6 h on a 24 GB
card. [`docs/RUNBOOK.md`](docs/RUNBOOK.md) covers preflight failures and what each one means.

Two settings matter more than they look. `SLMEVAL_DISABLE_THINKING` keeps hybrid-reasoning
models from spending hundreds of tokens per call on a chain of thought for the same answer.
`SLMEVAL_MAX_TOKENS` must be identical for every judge, or the legs stop being comparable — if
one model truncates, raise it for all of them and report the change.

## Limitations and future work

- **Masking is best-effort.** Presidio's NER misses entities, and only the configured entity
  types are masked at all. Indirect identifiers (a rare job title plus a city) survive
  masking. Treat `mask` as risk reduction, not anonymization, and keep an in-process judge
  when a miss would be unacceptable.
- **Results served by the API stay masked.** The placeholder mapping is never persisted, so
  the de-anonymized text exists only inside the process that did the masking. The CLI shows
  restored verdicts; the service does not.
- **The judge is the measurement instrument.** Faithfulness is what an SLM believes the
  contexts support, and small models are weaker at multi-hop and numeric reasoning. Always
  report the judge model and `k` alongside any score.
- **Response-level labels.** RAGTruth annotates spans; this harness collapses them to one
  boolean per response, which cannot distinguish one bad clause from a wholly fabricated
  answer.
- **Two metrics.** Faithfulness and relevance only — no context precision/recall, no answer
  completeness.
- **Not tuned.** No prompt or threshold was fitted to any benchmark example. The analysis
  prints two tables: an in-sample one that picks its threshold on the rows it reports, and a
  held-out one that picks on half the labelled rows and measures on the other. **Quote the
  held-out numbers** — the gap between the two is exactly how much the in-sample threshold
  flatters the judge.
- **Judges are compared on different subsets.** Because coverage differs (106 to 149 of 150),
  each judge's metrics come from the rows it could actually score. The counts are printed with
  every table for this reason; a judge that scores fewer, easier rows is not directly
  comparable to one that scores all of them.
- **One dataset, one host, no confidence intervals.** 150 English RAGTruth samples on a single
  RTX 4090. The held-out folds are small, so a 0.002 F1 difference between two judges is not a
  stable ranking, and none is claimed.
- **Deployment is verified by configuration, not by a live start.** The Compose topology is
  exercised by tests, but the rented benchmark host withheld the kernel capabilities Docker
  needs (`CAP_NET_ADMIN`), so a runtime `docker compose up` was not demonstrated there. It
  needs no GPU and is the one open item.
- Future work: span-level evaluation, confidence intervals, a batched judge API for throughput,
  comparison across quantization levels rather than only Q4_K_M, and non-English detector
  coverage.
