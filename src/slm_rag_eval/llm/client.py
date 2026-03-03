"""LLM client abstraction.

The concrete OpenAI-compatible client and the `generate_json` structured-output
helper are implemented in task M01 (tasks/todo/M01-llm-client.md). This module
pins the interface so tests and metrics can be written against it today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    """Normalized response from any backend."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    model: str = "unknown"


class LLMClient(Protocol):
    """Minimal async interface every backend (and the test fake) implements."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...
