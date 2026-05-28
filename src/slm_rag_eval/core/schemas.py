"""Core data models shared across the pipeline."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field


def _require_content(text: str) -> str:
    """Reject a string that carries no characters a judge could have reasoned about.

    A blank claim, question, or reason is schema-valid but semantically empty, and every
    aggregation downstream weights it exactly like a real one — a blank claim marked
    `supported` scores 1.0. Rejecting it here puts the repair loop to work instead.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("must not be blank")
    return stripped


def _require_distinct(values: list[str]) -> list[str]:
    """Reject repeated entries, comparing the way the alignment check does.

    Duplicates are not harmless: `["X", "X"]` gives claim X twice the weight of any other
    claim in the faithfulness average, so a judge that repeats itself silently reweights
    the score.
    """
    seen: set[str] = set()
    for value in values:
        key = " ".join(value.split()).casefold()
        if key in seen:
            raise ValueError(f"must not repeat an entry: {value!r} appears more than once")
        seen.add(key)
    return values


NonBlankStr = Annotated[str, AfterValidator(_require_content)]
"""A string a judge actually filled in, stripped of surrounding whitespace."""

DistinctNonBlankList = Annotated[list[NonBlankStr], AfterValidator(_require_distinct)]
"""A list of non-blank strings with no repeats — the shape every judge list output needs."""


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
