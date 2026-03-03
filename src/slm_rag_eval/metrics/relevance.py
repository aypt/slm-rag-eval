"""Answer-relevance metric. Implemented in task M03 (tasks/todo/M03-relevance-registry.md)."""

from __future__ import annotations

from slm_rag_eval.core.schemas import EvalRequest, EvalResult
from slm_rag_eval.llm.client import LLMClient


async def score_relevance(request: EvalRequest, judge: LLMClient) -> EvalResult:
    raise NotImplementedError("Implemented in task M03 — see tasks/todo/M03-relevance-registry.md")
