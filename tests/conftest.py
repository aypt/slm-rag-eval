"""Shared fixtures. FakeLLMClient is the backbone of every unit test: no network, ever."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from slm_rag_eval.llm.client import LLMResponse


@dataclass
class RecordedCall:
    """One captured judge call, for assertions (including the privacy invariant in M04)."""

    messages: list[dict[str, Any]]
    json_schema: dict[str, Any] | None
    temperature: float
    max_tokens: int

    def full_text(self) -> str:
        """All message content concatenated — convenient for 'PII never leaves' checks."""
        return "\n".join(str(m.get("content", "")) for m in self.messages)


@dataclass
class FakeLLMClient:
    """Scripted stand-in for a judge backend.

    Usage: push scripted response texts, run code under test, then assert on `.calls`.
    Raises if the code under test asks for more responses than were scripted, which
    keeps tests honest about how many judge calls a code path makes.
    """

    responses: list[str] = field(default_factory=list)
    calls: list[RecordedCall] = field(default_factory=list)

    def push(self, *texts: str) -> None:
        self.responses.extend(texts)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(
            RecordedCall(
                messages=messages,
                json_schema=json_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if not self.responses:
            raise AssertionError(
                "FakeLLMClient ran out of scripted responses; push() more in the test."
            )
        return LLMResponse(text=self.responses.pop(0), model="fake")


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()
