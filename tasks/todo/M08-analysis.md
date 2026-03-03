# M08 · Benchmark analysis + figures (report material)

Context: repo slm-rag-eval; M07 done. Input files are the JSONL rows produced by
bench.run — each row: {sample_id, dataset, judge, model, label_hallucinated,
scores: {faithfulness, relevance?}, verdicts, latency_ms, usage}. This file is the
complete spec; also follow AGENTS.md.

Task: implement src/slm_rag_eval/bench/analyze.py with CLI
`python -m slm_rag_eval.bench.analyze results/*.jsonl --out report/`.

Requirements:
1. Detection quality vs human labels, per judge: predicted_hallucinated :=
   faithfulness < t; sweep t over [0,1] step 0.05; report precision, recall, F1,
   balanced accuracy, ROC-AUC; select and report the best-F1 threshold per judge.
   Rows with faithfulness None are counted separately as "unscored" and excluded
   from threshold metrics.
2. SLM-vs-cloud agreement (when both judges cover the same sample_ids): Pearson and
   Spearman correlation on faithfulness scores; Cohen's kappa on binary decisions
   at each judge's own best threshold.
3. Efficiency, per judge: median and p95 latency per sample; mean tokens per
   sample; estimated cost table (USD per 1M input/output tokens for the cloud judge
   configurable via CLI flags; local judge reported as 0 marginal cost with a
   footnote about hardware).
4. Outputs: report/summary.md containing all tables, plus PNG figures via
   matplotlib: ROC curves (one figure, one line per judge), faithfulness score
   distributions per judge, latency box plot. matplotlib only, default colors, one
   chart per figure.
5. New deps in an "analysis" optional-dependency group: numpy, pandas,
   scikit-learn, matplotlib. `make setup` installs it.

Testing: run analyze on a synthetic fixture JSONL with hand-computed expected
values; assert exact metric numbers (within 1e-9) and that every output file
exists; a two-judge fixture exercises the agreement stats.

Definition of done: `make check` green; a sample report generated from the
synthetic fixtures is committed under docs/sample_report/ for illustration;
PROGRESS.md updated.
