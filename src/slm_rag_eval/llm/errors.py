"""Exceptions raised by the LLM inference layer."""

from __future__ import annotations


class JSONGenerationError(RuntimeError):
    """Raised when an LLM does not produce schema-valid JSON within the retry limit."""

    def __init__(self, message: str, *, last_raw_text: str) -> None:
        super().__init__(message)
        self.last_raw_text = last_raw_text
