"""PII sanitization layer built on Microsoft Presidio.

Implemented in task M04 (tasks/todo/M04-privacy-layer.md).
"""

from __future__ import annotations


class Sanitizer:
    """Detects PII and replaces it with stable placeholders before any judge call."""

    def __init__(self) -> None:
        raise NotImplementedError("Implemented in task M04 — see tasks/todo/M04-privacy-layer.md")
