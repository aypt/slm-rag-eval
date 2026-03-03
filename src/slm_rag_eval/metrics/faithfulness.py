"""Faithfulness metric: claim extraction + per-claim verification against contexts.

Implemented in task M02 (tasks/todo/M02-faithfulness.md).
"""

from __future__ import annotations

from slm_rag_eval.core.schemas import EvalRequest, EvalResult
from slm_rag_eval.llm.client import LLMClient


async def score_faithfulness(request: EvalRequest, judge: LLMClient) -> EvalResult:
    raise NotImplementedError("Implemented in task M02 — see tasks/todo/M02-faithfulness.md")
