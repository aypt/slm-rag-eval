"""Core data models shared across the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    """One RAG interaction to evaluate."""

    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    """Verdict for a single atomic claim extracted from the answer."""

    claim: str
    verdict: Literal["supported", "unsupported", "uncertain"]
    reason: str


class EvalResult(BaseModel):
    """Aggregated evaluation output for one EvalRequest."""

    faithfulness: float | None = None
    relevance: float | None = None
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    model_info: dict[str, object] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
