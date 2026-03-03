# M01 · LLM client abstraction

Context: repo slm-rag-eval. This file is the complete spec; also follow AGENTS.md.
The interface (LLMClient Protocol, LLMResponse) is already pinned in
src/slm_rag_eval/llm/client.py — extend that module, keep the existing names.

Task: implement the inference layer in src/slm_rag_eval/llm/.

Requirements:
1. OpenAICompatClient implementing LLMClient using httpx.AsyncClient against any
   OpenAI-compatible /v1/chat/completions endpoint (works for Ollama, vLLM, OpenAI).
   Configuration comes from core.config.Settings (env prefix SLMEVAL_): base_url,
   api_key (optional; send Authorization header only when set), model, plus new
   fields timeout_s (default 120) and max_retries (default 3).
2. Fill LLMResponse properly: text from the first choice, usage prompt/completion
   token counts when the backend returns them, measured latency_ms, model name.
3. generate_json(client, messages, schema: type[BaseModel]) -> BaseModel in a new
   module src/slm_rag_eval/llm/structured.py:
   - asks for JSON output, parses response text (strip markdown fences if present),
     validates with the given Pydantic schema;
   - on parse/validation failure, retries up to 3 times, appending the error message
     to the conversation as a user turn ("repair loop");
   - raises JSONGenerationError (new exception in llm/errors.py) after exhausting
     retries, carrying the last raw text for debugging.
4. Transport retries with exponential backoff (tenacity) on 429/5xx/timeouts —
   distinct from and inside of the repair loop.
5. build_client(settings) -> LLMClient factory in llm/__init__.py so backends are
   swappable via config alone.

Testing (unit only, no real network):
- generate_json happy path, fence-stripping, repair loop success on 2nd try, and
  JSONGenerationError after 3 failures — all with the FakeLLMClient fixture.
- OpenAICompatClient against httpx.MockTransport: success, 429-then-success,
  timeout raising after retries, malformed body.

Definition of done: `make check` green; new-module coverage is meaningful (every
public function exercised); README gains a short "Configuring model backends"
section with an Ollama example (SLMEVAL_BASE_URL=http://localhost:11434/v1).
Append the acceptance checklist with evidence to PROGRESS.md.
