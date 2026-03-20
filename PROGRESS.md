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
