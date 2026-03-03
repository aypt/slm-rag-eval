"""Metric registry and single entry point for running evaluations.

Implemented in task M03; the privacy hook is wired in task M04.
"""

from __future__ import annotations

from slm_rag_eval.core.schemas import EvalRequest, EvalResult
from slm_rag_eval.llm.client import LLMClient


async def evaluate(
    request: EvalRequest,
    *,
    judge: LLMClient,
    metrics: list[str] | None = None,
    k: int = 1,
    strict: bool = True,
) -> EvalResult:
    raise NotImplementedError("Implemented in task M03 — see tasks/todo/M03-relevance-registry.md")
