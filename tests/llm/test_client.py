from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm import build_client
from slm_rag_eval.llm.client import OpenAICompatClient


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "base_url": "https://model.invalid/v1/",
        "api_key": "test-key",
        "model": "test-model",
        "timeout_s": 2,
        "max_retries": 3,
    }
    values.update(overrides)
    return Settings(**values)


async def test_openai_compat_client_success() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                "model": "served-model",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatClient(_settings(), http_client=http_client)
        response = await client.complete(
            [{"role": "user", "content": "Hi"}],
            json_schema={"title": "Greeting", "type": "object"},
            temperature=0.2,
            max_tokens=20,
        )

    assert response.text == "hello"
    assert response.usage == {"prompt_tokens": 7, "completion_tokens": 2}
    assert response.latency_ms >= 0
    assert response.model == "served-model"
    assert str(requests[0].url) == "https://model.invalid/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 20
    assert payload["response_format"]["json_schema"]["schema"] == {
        "title": "Greeting",
        "type": "object",
    }


async def test_openai_compat_client_retries_429_then_succeeds() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "recovered"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatClient(_settings(api_key=None), http_client=http_client)
        response = await client.complete([{"role": "user", "content": "Hi"}])

    assert attempts == 2
    assert response.text == "recovered"
    assert response.usage == {}
    assert response.model == "test-model"


async def test_openai_compat_client_raises_timeout_after_retries() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("model timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatClient(_settings(), http_client=http_client)
        with pytest.raises(httpx.ReadTimeout):
            await client.complete([{"role": "user", "content": "Hi"}])

    assert attempts == 3


async def test_openai_compat_client_rejects_malformed_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatClient(_settings(), http_client=http_client)
        with pytest.raises(ValueError, match="Malformed chat-completions response body"):
            await client.complete([{"role": "user", "content": "Hi"}])


async def test_build_client_uses_openai_compatible_backend() -> None:
    client = build_client(_settings())

    assert isinstance(client, OpenAICompatClient)
    await client.aclose()
