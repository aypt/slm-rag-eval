"""Exceptions raised by the LLM inference layer."""

from __future__ import annotations


class JSONGenerationError(RuntimeError):
    """Raised when an LLM does not produce schema-valid JSON within the retry limit."""

    def __init__(self, message: str, *, last_raw_text: str) -> None:
        super().__init__(message)
        self.last_raw_text = last_raw_text


class TruncatedResponseError(ValueError):
    """Raised when a backend stopped at the token budget instead of finishing its JSON.

    Distinct from a malformed body on purpose: truncation is a budget problem the operator
    fixes by raising `SLMEVAL_MAX_TOKENS`, whereas a malformed body is a backend problem.
    Collapsing the two hides the one that has an easy fix, and long RAGTruth contexts hit
    it often enough that the difference matters when reading a failure list.
    """


class JudgeRefusalError(ValueError):
    """Raised when a backend returned an explicit refusal instead of content.

    A refusal is a real, reportable evaluation outcome — not a parse failure — so it keeps
    its own type and its message carries whatever the model said.
    """
