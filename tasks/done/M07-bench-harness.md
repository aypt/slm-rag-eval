# M07 · Benchmark harness (datasets + runner)

Context: repo slm-rag-eval; M01–M06 done. This file is the complete spec; also
follow AGENTS.md.

Task: build the benchmark harness in src/slm_rag_eval/bench/.

Requirements:
1. Common sample type LabeledSample{id, question, answer, contexts: list[str],
   label_hallucinated: bool, meta: dict} and loader interface
   load(name: str, limit: int | None = None) -> list[LabeledSample] in datasets.py.
2. RAGTruth loader (primary): scripts/download_ragtruth.py fetches the official
   dataset from its GitHub repository into data/ragtruth/ (gitignored); the loader
   converts response-level annotations to label_hallucinated. Record the dataset's
   license and citation in README. The loader works offline once files are cached
   and raises a clear error telling the user to run the download script otherwise.
3. HaluEval QA-subset loader (secondary), same pattern
   (scripts/download_halueval.py).
4. Runner CLI `python -m slm_rag_eval.bench.run --dataset ragtruth --judge slm
   --limit 200 --out results/` implemented with typer in run.py:
   - judge "slm" builds the client from Settings (Ollama); judge "cloud" builds it
     from CLOUD_BASE_URL / CLOUD_API_KEY / CLOUD_MODEL env vars (any
     OpenAI-compatible API).
   - Add a prominent README warning: the cloud judge is for PUBLIC benchmark data
     only — never send private/production data to it; that is the entire point of
     this project.
   - Writes one JSONL row per sample: {sample_id, dataset, judge, model,
     label_hallucinated, scores: {faithfulness, relevance?}, verdicts, latency_ms,
     usage} plus a run-manifest JSON next to it (params, git sha, UTC timestamp,
     failure count).
   - Resumable: rerunning with the same --out file skips sample_ids already present.
   - --privacy-mode flag passed through to the pipeline (default from Settings).
   - Fail soft: an exception on one sample logs it, counts it in the manifest, and
     continues.

Testing: loaders tested against tiny hand-written fixture files in tests/data
(5 rows, fully synthetic text — do NOT copy real dataset content into fixtures);
runner end-to-end with FakeLLMClient writing to tmp_path, including resume behavior
and the fail-soft path.

Definition of done: `make check` green; `python -m slm_rag_eval.bench.run --help`
documented in README with an example SLM-vs-cloud comparison workflow;
PROGRESS.md updated.
