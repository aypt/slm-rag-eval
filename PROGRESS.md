# PROGRESS

Append-only evidence log. Every completed task adds a section here: each acceptance
criterion checked off with concrete evidence (test names, command output).

## M00 · Scaffold — DONE
- [x] `make check` green (ruff + mypy + pytest) — verified locally at scaffold time.
- [x] Package imports; `/healthz` returns 200 (tests/service/test_api.py::test_healthz).
- [x] FakeLLMClient fixture records calls and enforces scripted responses
      (tests/test_fake_llm.py, 2 tests).
- [x] CI workflow present (.github/workflows/ci.yml) running `make check`.

## M01 · LLM client abstraction — DONE
- [x] `OpenAICompatClient` posts OpenAI chat-completions requests with settings-driven URL,
      model, timeout, optional bearer auth, and structured response format
      (`tests/llm/test_client.py::test_openai_compat_client_success`).
- [x] Responses normalize first-choice text, prompt/completion usage, measured latency, and
      backend model name (`tests/llm/test_client.py::test_openai_compat_client_success`).
- [x] Transient transport failures use bounded exponential-backoff retries independently of
      structured repairs (429 recovery and exhausted timeout tests in `tests/llm/test_client.py`).
- [x] Malformed backend bodies fail explicitly
      (`tests/llm/test_client.py::test_openai_compat_client_rejects_malformed_body`).
- [x] `generate_json` passes the Pydantic schema, strips Markdown fences, repairs invalid output,
      stops after three failed attempts, and preserves the final raw text in `JSONGenerationError`
      (`tests/llm/test_structured.py`, 4 tests using `FakeLLMClient`).
- [x] `build_client(settings)` selects the OpenAI-compatible backend
      (`tests/llm/test_client.py::test_build_client_uses_openai_compatible_backend`).
- [x] README documents `SLMEVAL_*` backend configuration with the required local Ollama URL.
- [x] `make check` green: ruff and mypy passed; pytest reported 30 passed.

## M02 · Faithfulness metric (core engine) — DONE
- [x] Two-stage claim extraction and context verification use `generate_json` with `Claims`
      and index-aligned `ClaimVerdict` batch schemas (`tests/metrics/test_faithfulness.py`).
- [x] Extraction prompt has exactly two few-shot examples, all required atomicity/filtering/
      verbatim-copy rules, and has a compact rendering comfortably below the approximate
      1,200-token budget in the snapshot test
      (`test_rendered_prompts_match_snapshots`; snapshots committed under
      `tests/metrics/snapshots/`).
- [x] Verification batches at most five claims (seven claims produce two verification calls),
      and a valid-but-wrong response length is re-asked once
      (`test_seven_claims_are_verified_in_batches_of_five`,
      `test_verdict_count_mismatch_is_reasked_once`).
- [x] Strict and non-strict scoring, an all-uncertain non-strict result, empty answers, and the
      zero-claim extraction retry are covered by dedicated `FakeLLMClient` tests.
- [x] `k=3` self-consistency majority voting and three-way tie-to-uncertain behavior are covered
      by `test_k_three_majority_vote_and_three_way_tie`.
- [x] Results include per-stage millisecond timings plus model, `k`, strictness, and aggregate
      prompt/completion/total token usage metadata.
- [x] `python examples/faithfulness_demo.py` runs without a network or model and prints a full
      `EvalResult` (score 0.5, two verdicts, model metadata, token totals, and timings).
- [x] `make check` green: ruff and mypy passed; pytest reported 43 passed.

## M03 · Relevance metric + metric registry — DONE
- [x] Relevance uses one `generate_json` invocation with `GeneratedQuestions` constrained to
      exactly three questions, followed by one invocation with three index-aligned 0–2 ratings
      and required reasons (`tests/metrics/test_relevance.py`).
- [x] Relevance is the mean normalized rating, records per-stage latency and aggregate token
      metadata, and returns `None` without a judge call for an empty answer
      (`test_relevance_generates_three_questions_then_rates_them`,
      `test_empty_answer_returns_none_without_calling_judge`).
- [x] The registry defaults to faithfulness, runs either supported metric, merges both scores
      and faithfulness verdicts, and namespaces per-metric latency and model/token metadata
      (`tests/metrics/test_registry.py::test_registry_merges_both_metrics_into_complete_result`,
      `test_registry_defaults_to_faithfulness`).
- [x] Unknown metric names fail before any judge call and list both available metrics
      (`test_registry_rejects_unknown_metric_before_judge_call`).
- [x] `Settings` provides the requested metric, self-consistency, strictness, and privacy
      defaults; JSON-list, integer, boolean, and `off` environment overrides are exercised in
      `tests/core/test_config.py`.
- [x] README documents both prompting approaches, scoring and failure modes, registry behavior,
      all configuration defaults, and the temporary M04 privacy-mode behavior.
- [x] Ambiguity decision: the rating response is represented as `QuestionRatings.ratings`, an
      exactly-three-item list aligned by generated-question index; this is the smallest
      structured schema that carries each numeric rating and its required reason.
- [x] `make check` green: ruff and mypy passed; pytest reported 52 passed.

## M04 · Privacy layer (Presidio PII sanitization) — DONE
- [x] Runtime dependencies include `presidio-analyzer`, `presidio-anonymizer`, and `spacy`;
      `make setup` downloads `en_core_web_lg`, whose NER role is documented in README.
- [x] Settings exposes the seven default Presidio entities and threshold 0.4, with JSON-list and
      numeric environment overrides (`tests/privacy/test_settings.py`, 2 tests).
- [x] `Sanitizer.sanitize` detects the seven configured default entity types at threshold 0.4,
      supports injected analyzers and custom entity/threshold settings, assigns collision-safe
      stable placeholders across a batch, and returns only the in-memory reverse mapping
      (`tests/privacy/test_sanitizer.py::test_sanitize_uses_stable_batch_placeholders_and_forwards_configuration`,
      `test_sanitize_avoids_colliding_with_existing_placeholder_text`).
- [x] `restore` performs exact, single-pass de-anonymization without cascading placeholder
      replacements (`tests/privacy/test_sanitizer.py::test_restore_is_exact_and_does_not_cascade_replacements`).
- [x] `metrics.registry.evaluate` honors `Settings.privacy_mode`, sanitizes question, answer, and
      all contexts as one batch before either metric can call the judge, bypasses sanitization
      only in explicit `off` mode, and restores display verdicts only after all judge calls
      (`tests/privacy/test_privacy_invariant.py`, 2 tests).
- [x] The non-negotiable `test_privacy_invariant_no_pii_leaves_process` uses an injected stub
      detector and asserts that no synthetic name, email, or phone substring appears in any of
      the four prompts captured by `FakeLLMClient.calls`; it runs independently of Presidio/model
      availability and carries the required do-not-weaken comment.
- [x] Real Presidio detection of a synthetic email and reserved fictional phone number is covered
      by `test_presidio_detects_synthetic_email_and_phone`, with an explicit model-unavailable
      skip reason; it passed in the verification environment with `en_core_web_lg` installed.
- [x] README's "Privacy layer" section documents the judge-boundary threat model, in-memory-only
      mapping, NER misses and other non-protections, mask/off behavior, default entities, threshold
      tradeoff, environment configuration, and why the spaCy model is required.
- [x] `Settings.privacy_entities` reuses `privacy.sanitizer.DEFAULT_ENTITIES` rather than repeating
      the seven entity names, so the default list has exactly one definition.
- [x] `make check` green **in the interpreter the runner gates on**
      (`PATH=.venv/bin:$PATH make check`): ruff passed, mypy reported no issues in 23 source
      files, and pytest reported 60 passed in 6.04 seconds.

## M04 · Environment root cause (why this task first landed in needs-review)
The runner filed M04 as `needs-review — check failed` twice
(`logs/M04-privacy-layer-20260803-102628.log`). The code was not at fault; two Python
environments were:

- `scripts/run_overnight.sh` puts `.venv/bin` on PATH, so the runner's `make check` gate ran
  under `.venv/bin/python`.
- `codex exec` runs its commands in `/bin/bash -lc` — a login shell that re-reads the profile
  and drops that PATH entry — so the agent's `make setup` installed `presidio-analyzer 2.2.364`
  and `en_core_web_lg 3.8.0` into `~/.local/lib/python3.12/site-packages` instead.
- `.venv/pyvenv.cfg` sets `include-system-site-packages = false`, so the venv could never see
  them. The agent's own `make check` was genuinely green; the runner's was genuinely red, with
  `ModuleNotFoundError: No module named 'presidio_analyzer'` raised through the fail-closed
  default mask path in the two (unmodified) M03 registry tests.

Fixes applied while restoring the work:
- [x] The dependencies were installed into `.venv` through the documented path
      (`PATH=.venv/bin:$PATH make setup`), and `.venv/bin/python -c "import presidio_analyzer,
      en_core_web_lg"` now succeeds.
- [x] A `$PWD`-guarded snippet in `~/.profile` (outside the repo — no repo file changed) puts
      this repo's `.venv/bin` on PATH for login shells started inside it, so agent shells and the
      runner now share one interpreter. Verified: `bash -lc 'which python'` in the repo resolves
      to `.venv/bin/python`, while shells started elsewhere are unaffected.
- [x] `.github/workflows/ci.yml` installs with `make setup` instead of a bare
      `pip install -e ".[dev]"`. The Presidio packages alone are not enough — `AnalyzerEngine()`
      needs the `en_core_web_lg` model at construction time, so CI would otherwise have gone red
      on the same fail-closed path. The `make check` gate step itself is unchanged.
- [x] Known cost, accepted rather than engineered around: the suite went from ~0.9 s to ~6 s
      because the two M03 registry tests now load the spaCy model once through the default
      sanitizer. Making them fast again would mean editing existing tests to inject a no-op
      sanitizer (forbidden) or weakening the `privacy_mode="mask"` default (the wrong trade).

Open TODO for M05: `registry.evaluate` masks internally and returns only an `EvalResult`, so the
sanitized request is not visible to callers. M05 requirement 4 ("persist the sanitized request,
never the raw one") therefore needs `evaluate` to also hand back the masked `EvalRequest` it sent
to the judge — extend the return or accept a pre-sanitized request; do not re-run the detector a
second time in the worker, and never persist the placeholder mapping.

## M05 · API service + async worker + persistence — DONE
- [x] `POST /v1/evaluations` accepts an `EvalRequest` plus `options{metrics,k,strict,privacy_mode}`,
      creates a queued `Job`, and returns 202 `{job_id, status:"queued"}`; `GET
      /v1/evaluations/{job_id}` returns status with `result`/`error`, and 404 for an unknown id
      (`tests/service/test_evaluations_api.py::test_submit_then_embedded_worker_completes_the_job`,
      `test_submitted_options_override_settings_defaults`, `test_unknown_job_returns_404`).
      `GET /healthz` is unchanged (`tests/service/test_api.py::test_healthz`).
- [x] `service/db.py` maps `Job{id,status,request_json,result_json,error,created_at,updated_at}`
      with SQLAlchemy 2 async; `create_engine` makes the SQLite parent directory, `DATABASE_URL`
      overrides the `sqlite+aiosqlite:///data/jobs.db` default (Postgres-ready), and tables are
      created in the app lifespan.
- [x] Job claiming is a conditional `UPDATE ... WHERE status='queued'` whose `rowcount` decides
      the winner — no `SELECT FOR UPDATE`, so it is correct for single-writer SQLite and for
      Postgres (`test_queued_job_is_claimed_exactly_once`).
- [x] The worker runs jobs under an `asyncio.Semaphore(WORKER_CONCURRENCY, default 2)`, retries a
      failed job exactly once and then stores the message with status `error`
      (`test_failing_judge_marks_the_job_error_after_one_retry` asserts the judge saw exactly two
      attempts). It runs embedded via lifespan when `WORKER_EMBEDDED=1` (default) or standalone
      via `python -m slm_rag_eval.service.worker`.
- [x] Privacy at rest: masking happens at the API boundary before the row is written, so the job
      row, the judge prompts, and the stored result are all placeholder-only
      (`tests/service/test_privacy_at_rest.py::test_privacy_invariant_no_pii_is_persisted_or_sent`,
      which must never be weakened). `privacy_mode="off"` stores the submission unchanged
      (`test_privacy_mode_off_persists_the_request_unchanged`).
- [x] `registry.sanitize_for_judge` exposes the masked request that `evaluate` would send, which
      is what lets the API persist the sanitized form without running the detector twice — the
      open TODO recorded in the M04 section is now closed.
- [x] Tests drive the app through `httpx.ASGITransport` with the lifespan running, the judge
      injected via `create_app(judge_factory=...)`, and a bounded poll loop
      (`tests/service/factories.py`); no test touches the network or a real model.
- [x] Live check of the DoD: `make run`, then `curl /healthz` → `{"status":"ok"}`;
      `POST /v1/evaluations` → 202 with a job id; `GET` → settled job; unknown id → 404. With the
      default `privacy_mode="mask"` and real Presidio, the persisted `request_json` read straight
      out of `data/jobs.db` was `"<PERSON_1> filed the ticket from <EMAIL_ADDRESS_1>."` with no raw
      value anywhere in the row. (The job itself ended `error: All connection attempts failed`,
      which is correct: no Ollama judge is running in this container.)
- [x] Ambiguity decisions (AGENTS.md rule 7): (a) the spec's `Job` column list has no options
      column, so the resolved options are stored inside `request_json` as
      `{"request": ..., "options": ...}`; (b) the worker evaluates the already-masked stored
      request with `privacy_mode="off"` rather than masking a second time — re-running the
      detector over text that already contains `<PERSON_1>` placeholders risks corrupting them,
      and the boundary has already been applied.
- [x] `make check` green: ruff and mypy passed (24 source files); pytest reported 67 passed.

Open TODO for M06: `.env.example` does not exist yet; M06 requirement 3 creates it and must cover
`DATABASE_URL`, `WORKER_EMBEDDED`, and `WORKER_CONCURRENCY` alongside the `SLMEVAL_*` settings.

## M06 · Containerization (Dockerfile + docker-compose) — DONE
- [x] Multi-stage `Dockerfile` on `python:3.11-slim`: the builder installs `.[postgres]` plus
      `en_core_web_lg`, and the runtime stage copies only `site-packages` and `/usr/local/bin`,
      so `build-essential` never reaches the runtime image. It runs as the non-root `slmeval`
      user and HEALTHCHECKs `/healthz` with `urllib` instead of adding `curl`
      (`tests/deploy/test_deployment_config.py::test_dockerfile_is_multi_stage_slim_non_root_and_healthchecked`).
- [x] `docker-compose.yml` defines `api` (`WORKER_EMBEDDED=0`, port 8000), `worker`
      (`python -m slm_rag_eval.service.worker`), `db` (`postgres:16`, named volume,
      `pg_isready` healthcheck, both app services waiting on `service_healthy`), `ollama`
      (named model volume, commented GPU `deploy.resources.reservations.devices` block), and
      the one-shot `ollama-init` that pulls `JUDGE_MODEL` and exits
      (`test_compose_defines_the_full_stack`, `test_api_serves_http_while_the_worker_runs_separately`,
      `test_database_is_postgres_with_a_readiness_check_and_named_volume`,
      `test_ollama_init_pulls_the_configured_judge_model`,
      `test_gpu_reservation_block_is_present_but_commented_out`).
- [x] `.env.example` documents every setting; `test_env_example_documents_every_setting` derives
      the expected variable names from `Settings.model_fields` (including the aliased
      `DATABASE_URL`, `WORKER_EMBEDDED`, `WORKER_CONCURRENCY`) plus `JUDGE_MODEL`, so the file
      cannot silently drift from the model. This closes the open TODO left by M05.
- [x] Make targets `docker-build`, `docker-up`, `docker-down`, `docker-logs` added; the
      `check`/`lint`/`type`/`test` targets are untouched.
- [x] CI gains a `docker` job that runs `docker build` and, after `cp .env.example .env`,
      `docker compose config --quiet`. Build only — no compose run, since CI has no judge model.
- [x] New runtime extra `[postgres]` (`asyncpg`) because the compose `DATABASE_URL` is
      `postgresql+asyncpg://…`; a plain local install stays SQLite-only. `pyyaml` added to
      `[dev]` for the compose tests.
- [x] `.dockerignore` keeps the 900 MB `.venv`, caches, logs, and `data/` out of the build
      context.
- [x] Constraint honored: unit tests never need Docker — the deployment tests parse
      `docker-compose.yml` and the `Dockerfile` directly.
- [x] Not verified here, and deliberately not claimed: `docker build` and `docker compose config`
      could not be run in this dev container, which has no docker CLI and no
      `/var/run/docker.sock` (`which docker` → not found). The spec allows this; CI's `docker` job
      is what actually executes both, and the structural tests above are the local stand-in.
- [x] `make check` green: ruff and mypy passed (24 source files); pytest reported 74 passed.

## M07 · Benchmark harness (datasets + runner) — DONE
- [x] `LabeledSample{id,question,answer,contexts,label_hallucinated,meta}` and
      `load(name, limit=None, *, data_dir=None)` live in `bench/datasets.py`; unknown names and
      non-positive limits fail with the available datasets listed
      (`tests/bench/test_datasets.py::test_unknown_dataset_lists_the_available_ones`,
      `test_limit_truncates_and_must_be_positive`).
- [x] The RAGTruth loader joins `response.jsonl` with `source_info.jsonl` on `source_id`, maps
      RAGTruth's span annotations to a response-level `label_hallucinated` (any span ⇒ True), and
      normalizes all three task types: QA takes its question from `source_info` and splits the
      `passage N:` blob into separate contexts, Summary uses the prompt as the question and the
      document as the single context, and Data2txt keeps the structured record as one context
      (`test_ragtruth_loader_joins_sources_and_maps_span_labels`,
      `test_ragtruth_loader_normalizes_summary_and_structured_tasks`). Rows whose `source_id` has
      no source are dropped rather than half-loaded.
- [x] The HaluEval QA loader emits one non-hallucinated and one hallucinated sample per source
      row, with the knowledge field as the context
      (`test_halueval_loader_yields_one_positive_and_one_negative_per_row`).
- [x] Loaders are offline-only; a missing cache raises `DatasetNotDownloadedError` naming the
      download script (`test_missing_cache_names_the_download_script`).
      `scripts/download_ragtruth.py` and `scripts/download_halueval.py` are thin entry points over
      `bench/download.py`, which caches into `data/<name>/`, skips files already present unless
      `--force`, and writes through a `.part` file so an interrupted download is never cached
      (`tests/bench/test_download.py`, 3 tests using an injected fetcher — no test touches the
      network).
- [x] `python -m slm_rag_eval.bench.run` (typer) writes one JSONL row per sample with exactly
      `{sample_id, dataset, judge, model, label_hallucinated, scores, verdicts, latency_ms, usage}`
      plus a manifest holding params, git sha, UTC timestamp, and failure count
      (`tests/bench/test_run.py::test_run_writes_one_row_per_sample_plus_a_manifest`).
- [x] Resume works on the rows file: a rerun skips sample ids already present and only spends
      judge calls on new samples (`test_rerunning_the_same_output_skips_completed_samples`).
- [x] Fail-soft: a sample that raises is recorded in `manifest.failures` and the run continues
      (`test_one_failing_sample_does_not_end_the_run`).
- [x] `--judge slm` builds the client from `Settings`; `--judge cloud` builds it from
      `CLOUD_BASE_URL` / `CLOUD_MODEL` / `CLOUD_API_KEY` and fails with an explicit message when
      they are unset (`test_cloud_judge_requires_explicit_configuration`,
      `test_unknown_judge_is_rejected`). `--privacy-mode` overrides the configured mode.
- [x] README documents the download scripts, the run/resume workflow, `--help`, the SLM-vs-cloud
      comparison, both dataset licenses (MIT) and citations, and carries the prominent warning
      that the cloud judge is for PUBLIC benchmark data only.
- [x] Fixtures under `tests/data/` are hand-written and fully synthetic (5 HaluEval rows, 5
      RAGTruth responses over 3 sources); no real dataset text and no personal data.
- [x] Live end-to-end CLI check with no judge running:
      `python -m slm_rag_eval.bench.run --dataset halueval --limit 2 --privacy-mode off
      --data-dir tests/data --out <tmp>` loaded the fixtures, failed soft on both samples, and
      wrote a manifest with `failure_count: 2`, both `ConnectError` messages, and the real git sha.
- [x] Deviation from AGENTS.md rule 6, flagged deliberately: the spec requires new files under
      `scripts/`, which that rule otherwise puts off-limits. Both files are new, additive, and
      contain no runner logic; no existing file under `scripts/` was touched. A future runner
      invocation would trip its `scripts/ modified` smell check on them, which is why they are
      committed now rather than during a task run.
- [x] Ambiguity decision (AGENTS.md rule 7): `--out` is a directory, and the rows/manifest names
      are derived from the dataset and judge (`<dataset>_<judge>.jsonl`,
      `<dataset>_<judge>.manifest.json`). That is what makes "rerunning with the same `--out`"
      resumable while keeping the two judges' rows in separate files for M08 to compare.
- [x] `make check` green: ruff and mypy passed (25 source files); pytest reported 88 passed.

## M08 · Benchmark analysis + figures — DONE
- [x] Detection quality per judge: `predicted_hallucinated := faithfulness < t` swept over
      `[0,1]` in 0.05 steps (21 thresholds) with precision, recall, F1, and balanced accuracy at
      each step, plus threshold-independent ROC-AUC computed on the negated score (lower
      faithfulness ⇒ more likely hallucinated). Best-F1 threshold is reported per judge, ties
      resolved to the lower threshold
      (`tests/bench/test_analyze.py::test_slm_sweep_matches_hand_computed_confusion_counts`,
      `test_best_threshold_is_the_lowest_one_reaching_the_top_f1`,
      `test_roc_auc_is_perfect_when_scores_separate_the_classes`).
- [x] Rows with `faithfulness: null` are counted as `unscored` and excluded from every threshold
      metric — never coerced to 0.0 (`test_unscored_rows_are_counted_separately_and_excluded`;
      the fixture's `s5` exists precisely to catch that).
- [x] Agreement over samples both judges scored: Pearson and Spearman on faithfulness, and
      Cohen's kappa on the binary calls at each judge's own best threshold
      (`test_agreement_over_the_samples_both_judges_scored`). Pearson is asserted against the
      hand-derived expression `0.45 / sqrt(0.59 * 0.35)` to 1e-9; Spearman is exactly 1.0 because
      both judges rank the four shared samples identically.
- [x] Efficiency per judge: median and p95 latency (linear interpolation, documented in the
      report), mean tokens, and an estimated cost table driven by
      `--cloud-input-cost-per-1m` / `--cloud-output-cost-per-1m`; the local judge is 0 with a
      footnote about amortized hardware (`test_latency_and_token_efficiency_per_judge`,
      `test_cost_model_prices_the_cloud_judge_and_zeroes_the_local_one`).
- [x] `python -m slm_rag_eval.bench.analyze results/*.jsonl --out report/` writes `summary.md`
      with every table plus `roc_curves.png`, `score_distributions.png`, and `latency_box.png`
      — matplotlib only, default colors, one chart per figure
      (`test_analyze_writes_every_output_file`). Judges are additionally distinguished by line
      style on the ROC figure so identity does not depend on color alone.
- [x] All three figures were rendered and visually inspected: labeled axes, no label collisions,
      legend on the multi-series figure, medians matching the hand-computed 250 ms (cloud) and
      30 ms (slm).
- [x] New `[analysis]` optional dependency group (numpy, pandas, scikit-learn, matplotlib) and
      `make setup` now installs `.[dev,analysis]`. SciPy is deliberately not used: Pearson is
      `numpy.corrcoef` and Spearman is the same on pandas average ranks, so the analysis has no
      undeclared transitive dependency.
- [x] A sample report generated from the synthetic fixture is committed under
      `docs/sample_report/`, with a README note that its numbers are illustrative only.
- [x] `make check` green: ruff and mypy passed (25 source files); pytest reported 98 passed.

## M09 · Demonstration interface (CLI + Streamlit dashboard) — DONE
- [x] `rageval eval` accepts either `--question/--answer/--context` (repeatable) or `--json FILE`,
      prints an aligned per-claim table plus scores, judge model, privacy mode, and timings, and
      exits 1 only when faithfulness is below `--fail-under` (default 0.0, so it never fails
      unless asked) — `tests/test_cli.py::test_eval_prints_the_claim_table_and_scores`,
      `test_eval_reads_a_json_request_file`, `test_fail_under_sets_the_exit_code`,
      `test_fail_under_defaults_to_never_failing`,
      `test_eval_without_question_or_json_is_a_usage_error`.
- [x] `rageval batch FILE.jsonl --out results.jsonl` writes the bench.run row schema minus
      `label_hallucinated`, which a CLI batch has no way to know
      (`test_batch_writes_one_row_per_input_line`).
- [x] `rageval serve` starts uvicorn in-process against `service.api:app`.
- [x] Tests drive the CLI through `typer.testing.CliRunner` with a `FakeLLMClient` injected via
      the documented DI seam `cli.judge_factory`; no test touches the network.
- [x] `apps/dashboard.py` is a single Streamlit file with an Evaluate tab (sanitized preview with
      masked spans bolded, claim table, score metrics, latency, token totals) and a Results tab
      (loads a bench JSONL and renders the M08 summary and figures inline), with base URL, model,
      privacy mode, and metrics editable in the sidebar. No custom CSS.
- [x] All dashboard logic lives in the new pure module `slm_rag_eval/demo.py` and is unit-tested
      without Streamlit (`tests/test_demo.py`, 7 tests covering the sanitized preview in both
      privacy modes, placeholder highlighting, verdict/score flattening, latency and token
      summation, table truncation, and the batch row shape).
- [x] Streamlit rendering itself is not unit-tested, per the spec. It was verified manually
      twice: `streamlit run apps/dashboard.py --server.headless true` answered HTTP 200, and
      `streamlit.testing.v1.AppTest.from_file("apps/dashboard.py").run()` executed the whole
      script with `at.exception` empty, reporting the title, both tabs, and the sidebar inputs.
      That check is deliberately not in the suite: Streamlit lives in the optional `[demo]`
      extra, and `make check` must stay green without it.
- [x] `docs/demo.md` is a copy-paste walkthrough (setup → CLI → batch → benchmark → dashboard →
      service) containing a real captured `rageval eval` transcript. The transcript is pinned by
      `test_demo_transcript_matches_the_documented_output`, which regenerates it through the test
      runner and asserts the documented text still matches (timings normalized, since only they
      vary).
- [x] New `[demo]` optional dependency group (streamlit) and a `make demo` target.
- [x] `make check` green: ruff and mypy passed (26 source files); pytest reported 113 passed.
      `ruff check apps` is clean too, although the untouched `lint` target only covers src/tests.

## M10 · Documentation, hardening, reproducibility — DONE
- [x] README overhaul: problem statement, ASCII architecture diagram (client → API → sanitizer →
      job row → worker → metrics → judge → bench/report), local and Docker quickstarts, metrics
      explanation, benchmark reproduction steps, and a "Limitations and future work" section that
      states plainly what masking does not protect, that the reported best-F1 threshold is
      selected on the data it is reported for, and that no prompt or threshold was fitted to any
      benchmark example.
- [x] The configuration reference table is generated from the `Settings` model by
      `scripts/emit_config_table.py` and pasted into the README, so documented defaults cannot
      drift from the code; `tests/deploy/test_env_example_documents_every_setting` independently
      fails if `.env.example` stops covering a field. Field descriptions were added to `Settings`
      to make the generated table useful.
- [x] Docstrings added to every remaining public function/class (verified with an AST sweep over
      `src/`), and mypy tightened with `disallow_untyped_defs` + `disallow_incomplete_defs` for
      `slm_rag_eval.core.*`, `llm.*`, and `metrics.*`. This edits `[tool.mypy]`, which AGENTS.md
      rule 2 otherwise puts off-limits — flagged deliberately: the task requires it, and the
      change only ADDS strictness. Nothing existing was relaxed; `make check` is green under it.
- [x] Error-path review found a real defect: an unknown metric name was accepted with 202 and only
      failed later in the worker. `registry.validate_metrics` is now called on the submit path, so
      an impossible request is rejected with 400 before a job row exists
      (`tests/service/test_errors_and_logging.py::test_unknown_metric_is_mapped_to_400_with_the_request_id`).
- [x] Consistent exception → HTTP mapping: `ValueError` → 400, `JSONGenerationError` → 502,
      `SQLAlchemyError` → 503, each with `{detail, request_id}`; malformed bodies still 422 before
      any judge call (`test_malformed_bodies_are_rejected_before_any_work`).
- [x] Structured JSON logging (`service/logging.py`): one JSON object per line with
      `request_id`/`job_id` correlation from context vars, exception rendering, and extras folded
      in (4 tests). A middleware assigns a request id, honors a caller-supplied `x-request-id`,
      and echoes it on the response (`test_every_response_carries_a_request_id`); the worker binds
      `job_id` around each job. Log records deliberately carry path/status only — never request
      text, which may hold the PII masking removed.
- [x] `make reproduce` runs a bench + analysis into `report/repro/`, using the configured judge
      when `SLMEVAL_BASE_URL` answers and an offline stub otherwise, with built-in synthetic
      samples when no dataset is cached (`tests/bench/test_reproduce.py`, 6 tests). The stub marks
      every claim `uncertain` — scoring 0.0 across the board in strict mode — precisely so its
      output cannot be mistaken for an evaluation result; it prints that warning too.
- [x] `constraints.txt` pins the exact versions of all 107 installed packages, with the
      regeneration command in its header, and the README shows `-c constraints.txt`.
- [x] Fresh-clone path verified by actually executing it, not by inspection. In an empty
      directory: `git clone` (commit 9001cfa) → `python3 -m venv .venv` → `make setup` (exit 0,
      `en_core_web_lg` downloaded) → `make check` (ruff clean, mypy clean on 28 source files,
      **126 passed in 12.16s**) → `make reproduce` (4 samples, `failure_count: 0`,
      `report/repro/summary.md` written, with the default `privacy_mode: mask` exercising real
      Presidio). Nothing broke, so nothing needed fixing; the transcript is quoted verbatim in the
      README under "Verified fresh-clone walkthrough".
- [x] `make check` green: ruff and mypy passed (28 source files); pytest reported 126 passed.
