# M03 · Relevance metric + metric registry

Context: repo slm-rag-eval; M01–M02 done. This file is the complete spec; also
follow AGENTS.md.

Task:
1. Implement src/slm_rag_eval/metrics/relevance.py (replace the stub, keep
   score_relevance): the judge generates 3 questions that the given answer would
   directly answer (one generate_json call, schema
   GeneratedQuestions{questions: list[str]}), then one more call rates how well
   each generated question matches the original question on a 0–2 scale with a
   one-sentence reason; relevance = mean(rating / 2) in [0,1]. Empty answer -> None.
2. Implement src/slm_rag_eval/metrics/registry.py (replace the stub, keep the
   evaluate signature): runs the requested metrics ("faithfulness", "relevance"),
   merges their outputs into one EvalResult, collects per-metric latency and token
   usage into timings/model_info. Unknown metric name -> ValueError listing
   available metrics. Default metrics=["faithfulness"].
3. Extend core.config.Settings with: enabled_metrics (default ["faithfulness"]),
   k (default 1), strict (default True), privacy_mode (default "mask"; the actual
   sanitizer arrives in M04 — for now "mask" and "off" must both be accepted and
   behave identically, with a TODO note referencing M04). Document defaults in
   README.

Testing: unit tests with FakeLLMClient; a registry integration test producing a
complete EvalResult with both metrics; config override via environment variables
tested with monkeypatch.

Definition of done: `make check` green; README "Metrics" section documents both
metrics, their prompting approach, and failure modes; PROGRESS.md updated.
