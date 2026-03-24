"""Metric registry and single entry point for running evaluations."""

from __future__ import annotations

from time import perf_counter

from slm_rag_eval.core.schemas import EvalRequest, EvalResult
from slm_rag_eval.llm.client import LLMClient
from slm_rag_eval.metrics.faithfulness import score_faithfulness
from slm_rag_eval.metrics.relevance import score_relevance


async def _faithfulness_runner(
    request: EvalRequest,
    judge: LLMClient,
    *,
    k: int,
    strict: bool,
) -> EvalResult:
    return await score_faithfulness(request, judge, k=k, strict=strict)


async def _relevance_runner(request: EvalRequest, judge: LLMClient) -> EvalResult:
    return await score_relevance(request, judge)


_AVAILABLE_METRICS = ("faithfulness", "relevance")


async def evaluate(
    request: EvalRequest,
    *,
    judge: LLMClient,
    metrics: list[str] | None = None,
    k: int = 1,
    strict: bool = True,
) -> EvalResult:
    """Run selected metrics and merge their scores and metadata."""
    selected = metrics if metrics is not None else ["faithfulness"]
    unknown = sorted(set(selected) - set(_AVAILABLE_METRICS))
    if unknown:
        available = ", ".join(_AVAILABLE_METRICS)
        raise ValueError(
            f"Unknown metric(s): {', '.join(unknown)}. Available metrics: {available}"
        )

    merged = EvalResult()
    for metric_name in dict.fromkeys(selected):
        started = perf_counter()
        if metric_name == "faithfulness":
            result = await _faithfulness_runner(request, judge, k=k, strict=strict)
        else:
            result = await _relevance_runner(request, judge)
        latency_ms = (perf_counter() - started) * 1000

        if metric_name == "faithfulness":
            merged.faithfulness = result.faithfulness
            merged.verdicts = result.verdicts
        else:
            merged.relevance = result.relevance
        merged.timings[f"{metric_name}_ms"] = latency_ms
        merged.model_info[metric_name] = result.model_info

    return merged
