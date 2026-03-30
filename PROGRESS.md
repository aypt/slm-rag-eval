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
