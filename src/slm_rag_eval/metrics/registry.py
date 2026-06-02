"""Metric registry and single entry point for running evaluations."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Protocol

from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.core.schemas import EvalRequest, EvalResult
from slm_rag_eval.llm.client import LLMClient
from slm_rag_eval.metrics.faithfulness import score_faithfulness
from slm_rag_eval.metrics.relevance import score_relevance
from slm_rag_eval.privacy.sanitizer import SanitizedBatch, Sanitizer


class RequestSanitizer(Protocol):
    """Injectable masking boundary used by tests and alternate detectors."""

    def sanitize(self, texts: list[str]) -> SanitizedBatch:
        """Mask PII in a batch of texts, returning them plus the placeholder mapping."""
        ...

    def restore(self, text: str, mapping: dict[str, str]) -> str:
        """Put the original values back for trusted local display."""
        ...


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


def available_metrics() -> tuple[str, ...]:
    """Metric names `evaluate` accepts."""
    return _AVAILABLE_METRICS


def validate_metrics(metrics: list[str]) -> None:
    """Raise ValueError naming the unknown metrics, so callers can reject early."""
    unknown = sorted(set(metrics) - set(_AVAILABLE_METRICS))
    if unknown:
        available = ", ".join(_AVAILABLE_METRICS)
        raise ValueError(
            f"Unknown metric(s): {', '.join(unknown)}. Available metrics: {available}"
        )


def validate_metric_selection(metrics: list[str]) -> None:
    """Reject a selection that cannot produce a result, before any work starts.

    `validate_metrics` alone is not enough for a batch runner: it only rejects unknown
    names, and it runs per sample inside a fail-soft loop, so a typo shows up as every
    sample failing while the command still exits 0. An empty selection is worse — it
    succeeds and scores nothing.
    """
    if not metrics:
        available = ", ".join(_AVAILABLE_METRICS)
        raise ValueError(
            f"No metrics selected, so nothing would be scored. Available metrics: {available}"
        )
    validate_metrics(metrics)


@lru_cache(maxsize=8)
def _default_sanitizer(entities: tuple[str, ...], score_threshold: float) -> Sanitizer:
    return Sanitizer(entities=entities, score_threshold=score_threshold)


def build_sanitizer(settings: Settings) -> Sanitizer:
    """Return the shared detector for these settings (cached: the model load is expensive)."""
    return _default_sanitizer(
        tuple(settings.privacy_entities),
        settings.privacy_score_threshold,
    )


def sanitize_for_judge(
    request: EvalRequest,
    *,
    settings: Settings,
    sanitizer: RequestSanitizer | None = None,
) -> tuple[EvalRequest, dict[str, str]]:
    """Mask a request the way `evaluate` would, for callers that must persist the masked form.

    Returns the request as the judge will see it plus the placeholder mapping. The mapping is
    in-memory only: callers must never log or persist it (AGENTS.md rule 2).
    """
    if settings.privacy_mode != "mask":
        return request, {}
    return _sanitize_request(request, sanitizer or build_sanitizer(settings))


def _sanitize_request(
    request: EvalRequest,
    sanitizer: RequestSanitizer,
) -> tuple[EvalRequest, dict[str, str]]:
    batch = sanitizer.sanitize([request.question, request.answer, *request.contexts])
    expected_count = len(request.contexts) + 2
    if len(batch.texts) != expected_count:
        raise ValueError(
            f"Sanitizer returned {len(batch.texts)} texts for {expected_count} request fields"
        )
    sanitized_request = EvalRequest(
        question=batch.texts[0],
        answer=batch.texts[1],
        contexts=batch.texts[2:],
    )
    return sanitized_request, batch.mapping


def _restore_verdicts(
    result: EvalResult,
    sanitizer: RequestSanitizer,
    mapping: dict[str, str],
) -> None:
    for verdict in result.verdicts:
        verdict.claim = sanitizer.restore(verdict.claim, mapping)
        verdict.reason = sanitizer.restore(verdict.reason, mapping)


async def evaluate(
    request: EvalRequest,
    *,
    judge: LLMClient,
    metrics: list[str] | None = None,
    k: int = 1,
    strict: bool = True,
    settings: Settings | None = None,
    sanitizer: RequestSanitizer | None = None,
) -> EvalResult:
    """Mask a request, run selected metrics, and merge their scores and metadata."""
    selected = metrics if metrics is not None else ["faithfulness"]
    validate_metrics(selected)

    runtime_settings = settings if settings is not None else get_settings()
    active_sanitizer = sanitizer
    if runtime_settings.privacy_mode == "mask" and active_sanitizer is None:
        active_sanitizer = build_sanitizer(runtime_settings)
    judge_request, mapping = sanitize_for_judge(
        request,
        settings=runtime_settings,
        sanitizer=active_sanitizer,
    )

    merged = EvalResult()
    for metric_name in dict.fromkeys(selected):
        started = perf_counter()
        if metric_name == "faithfulness":
            result = await _faithfulness_runner(judge_request, judge, k=k, strict=strict)
        else:
            result = await _relevance_runner(judge_request, judge)
        latency_ms = (perf_counter() - started) * 1000

        if metric_name == "faithfulness":
            merged.faithfulness = result.faithfulness
            merged.verdicts = result.verdicts
        else:
            merged.relevance = result.relevance
        merged.timings[f"{metric_name}_ms"] = latency_ms
        merged.model_info[metric_name] = result.model_info

    if mapping and active_sanitizer is not None:
        _restore_verdicts(merged, active_sanitizer, mapping)
    return merged
