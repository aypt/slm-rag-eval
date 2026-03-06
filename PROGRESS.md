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
