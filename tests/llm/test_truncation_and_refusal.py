"""Truncation and refusal are defined outcomes, not malformed bodies.

Reporting either as "malformed chat-completions response body" blames the backend for
something the protocol specifies, and hides the one an operator can fix: a truncated
response just needs a bigger token budget. Long RAGTruth summarization contexts hit that
budget often enough that the distinction decides whether a failure list is readable.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm.client import OpenAICompatClient
from slm_rag_eval.llm.errors import JudgeRefusalError, TruncatedResponseError


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {"base_url": "http://judge.test/v1", "model": "stub"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _client(body: dict[str, Any], settings: Settings | None = None) -> OpenAICompatClient:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    resolved = settings or _settings()
    client = OpenAICompatClient(
        resolved, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    client.captured = captured  # type: ignore[attr-defined]  # test-only inspection hook
    return client


def _body(content: Any, finish_reason: str = "stop", **message_extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"content": content, **message_extra}
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "model": "stub",
    }


async def test_a_truncated_response_names_the_budget_and_the_setting() -> None:
    client = _client(_body('{"claims": ["half a cla', finish_reason="length"))

    with pytest.raises(TruncatedResponseError) as caught:
        await client.complete([{"role": "user", "content": "hi"}])

    message = str(caught.value)
    assert "2048" in message, "the operator needs to see which budget was hit"
    assert "SLMEVAL_MAX_TOKENS" in message


async def test_a_refusal_is_reported_as_a_refusal_and_carries_what_the_model_said() -> None:
    client = _client(_body(None, refusal="I cannot evaluate this content."))

    with pytest.raises(JudgeRefusalError, match="I cannot evaluate this content."):
        await client.complete([{"role": "user", "content": "hi"}])


async def test_a_genuinely_malformed_body_is_still_reported_as_malformed() -> None:
    client = _client({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]})

    with pytest.raises(ValueError, match="Malformed chat-completions response body"):
        await client.complete([{"role": "user", "content": "hi"}])


async def test_the_token_budget_comes_from_settings_rather_than_a_hardcoded_constant() -> None:
    client = _client(_body('{"ok": true}'), settings=_settings(max_tokens=4096))

    await client.complete([{"role": "user", "content": "hi"}])

    assert client.captured[0]["max_tokens"] == 4096  # type: ignore[attr-defined]


async def test_an_explicit_budget_still_wins_over_the_configured_one() -> None:
    client = _client(_body('{"ok": true}'), settings=_settings(max_tokens=4096))

    await client.complete([{"role": "user", "content": "hi"}], max_tokens=64)

    assert client.captured[0]["max_tokens"] == 64  # type: ignore[attr-defined]
