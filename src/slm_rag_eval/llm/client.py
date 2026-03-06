"""Backend-agnostic LLM interface and OpenAI-compatible implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from slm_rag_eval.core.config import Settings


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


def _is_retryable_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    return False


class OpenAICompatClient:
    """Client for an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._http_client = http_client or httpx.AsyncClient()
        self._owns_http_client = http_client is None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate one completion, retrying only transient transport failures."""
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            schema_name = str(json_schema.get("title") or "response")
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema},
            }

        headers: dict[str, str] = {}
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"

        endpoint = f"{self._settings.base_url.rstrip('/')}/chat/completions"
        started = perf_counter()
        response: httpx.Response | None = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=2.0),
            retry=retry_if_exception(_is_retryable_transport_error),
            reraise=True,
        ):
            with attempt:
                response = await self._http_client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self._settings.timeout_s,
                )
                response.raise_for_status()

        if response is None:  # Defensive: AsyncRetrying always makes at least one attempt.
            raise RuntimeError("The completion request was not attempted")
        latency_ms = (perf_counter() - started) * 1000
        return self._normalize_response(response, latency_ms)

    async def aclose(self) -> None:
        """Close the internally-created HTTP client, if this instance owns it."""
        if self._owns_http_client:
            await self._http_client.aclose()

    def _normalize_response(self, response: httpx.Response, latency_ms: float) -> LLMResponse:
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise TypeError("response body is not an object")
            choices = body["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError("response has no choices")
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                raise TypeError("first choice is not an object")
            message = first_choice["message"]
            if not isinstance(message, dict):
                raise TypeError("choice message is not an object")
            text = message["content"]
            if not isinstance(text, str):
                raise TypeError("choice content is not text")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed chat-completions response body") from exc

        usage: dict[str, int] = {}
        raw_usage = body.get("usage")
        if isinstance(raw_usage, dict):
            for key in ("prompt_tokens", "completion_tokens"):
                value = raw_usage.get(key)
                if isinstance(value, int):
                    usage[key] = value

        raw_model = body.get("model")
        model = raw_model if isinstance(raw_model, str) else self._settings.model
        return LLMResponse(text=text, usage=usage, latency_ms=latency_ms, model=model)
