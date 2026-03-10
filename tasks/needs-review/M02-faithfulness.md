# M02 · Faithfulness metric (core engine)

Context: repo slm-rag-eval; llm/ layer from M01 exists (OpenAICompatClient,
generate_json). This file is the complete spec; also follow AGENTS.md.

Task: implement src/slm_rag_eval/metrics/faithfulness.py (replace the stub, keep the
score_faithfulness entry point) using a two-stage, RAGAS-style approach optimized
for small (3–9B) instruct models.

Stage 1 — claim extraction: prompt the judge to decompose `answer` into atomic
factual claims; output validates against Claims{claims: list[str]} via generate_json.
Prompt requirements: exactly 2 few-shot examples; explicit rules (ignore opinions and
hedges, split conjoined facts into separate claims, copy numbers/dates/names
verbatim); keep the rendered prompt under ~1200 tokens.

Stage 2 — verification: for each claim, ask whether the provided contexts support
it; output items validating against ClaimVerdict (supported/unsupported/uncertain +
a one-sentence reason grounded in the context). Verify in batches of up to 5 claims
per call; the batch response schema is a list aligned by claim index — validate
length and re-ask once on mismatch.

Scoring: faithfulness = supported / total_claims, in [0,1]. Parameter strict
(default True): uncertain counts as unsupported; when False, uncertain is excluded
from the denominator (score None if all claims uncertain). Edge cases: empty answer
-> faithfulness None with an explanatory verdict list; zero extracted claims ->
retry extraction once, then None. Optional self-consistency: parameter k (default 1)
runs verification k times and majority-votes per claim (ties -> uncertain).

Populate EvalResult.timings (per-stage latency) and model_info (model name, k,
strict, token usage totals).

Testing: FakeLLMClient with scripted multi-call responses; cover scoring math,
batching (7 claims -> 2 calls), both strict modes, edge cases, k=3 voting including
a tie, and snapshot-test the exact rendered prompts (commit snapshots under
tests/metrics/snapshots/) so future prompt edits show up in diffs.

Definition of done: `make check` green; examples/faithfulness_demo.py runs against
FakeLLMClient and prints a full EvalResult; PROGRESS.md updated with the acceptance
checklist and evidence.
