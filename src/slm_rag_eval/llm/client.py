"""Backend-agnostic LLM interface and OpenAI-compatible implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm.errors import JudgeRefusalError, TruncatedResponseError


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
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate one completion, optionally constrained to a JSON schema.

        `max_tokens=None` means "use the configured budget", so the limit is one setting
        rather than a constant repeated at every call site.
        """
        ...


NO_THINK_DIRECTIVE = "/no_think"
"""Qwen3's documented switch for turning a hybrid-reasoning model's chain of thought off.

Applied at the prompt rather than as a request parameter on purpose. `reasoning.enabled`
is an OpenRouter gateway field that a local Ollama does not implement, whereas this
directive is handled by the chat template itself and therefore behaves the same on every
backend that serves the model. Measured on qwen3-8b: 339 completion tokens and 5.6 s become
47 tokens and 1.7 s, with identical extracted claims. A model without a thinking mode
ignores it — gemma-3-4b-it returned the same four claims either way.
"""


def apply_thinking_directive(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append the no-thinking directive to the final user turn, without mutating the input.

    Only the last user message is touched, and only once: repeating the directive in every
    turn would change the prompt a repair round sees relative to the first attempt.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") != "user":
            continue
        content = str(messages[index].get("content", ""))
        if NO_THINK_DIRECTIVE in content:
            return messages
        updated = list(messages)
        updated[index] = {**messages[index], "content": f"{content}\n{NO_THINK_DIRECTIVE}"}
        return updated
    return messages


def _raise_for_status_with_body(response: httpx.Response) -> None:
    """`raise_for_status`, but with the provider's explanation in the message.

    httpx reports only "Client error '400 Bad Request' for url …", while the body almost
    always says exactly what was wrong — a rejected schema, an unsupported parameter, an
    exhausted quota. Without it a 150-sample run fails 149 times and leaves a manifest of
    identical, uninformative errors, and finding the cause costs a separate investigation.
    """
    if response.is_success:
        return
    try:
        detail = response.text.strip()[:600]
    except (UnicodeDecodeError, httpx.ResponseNotRead):
        detail = ""
    message = (
        f"{response.status_code} {response.reason_phrase} from {response.request.url}"
        + (f": {detail}" if detail else "")
    )
    raise httpx.HTTPStatusError(message, request=response.request, response=response)


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
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate one completion, retrying only transient transport failures."""
        token_budget = max_tokens if max_tokens is not None else self._settings.max_tokens
        outbound = (
            apply_thinking_directive(messages) if self._settings.disable_thinking else messages
        )
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": outbound,
            "temperature": temperature,
            "max_tokens": token_budget,
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
                _raise_for_status_with_body(response)

        if response is None:  # Defensive: AsyncRetrying always makes at least one attempt.
            raise RuntimeError("The completion request was not attempted")
        latency_ms = (perf_counter() - started) * 1000
        return self._normalize_response(response, latency_ms, token_budget)

    async def aclose(self) -> None:
        """Close the internally-created HTTP client, if this instance owns it."""
        if self._owns_http_client:
            await self._http_client.aclose()

    def _normalize_response(
        self,
        response: httpx.Response,
        latency_ms: float,
        max_tokens: int,
    ) -> LLMResponse:
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
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed chat-completions response body") from exc

        # A refusal and a truncation are both well-formed responses that simply carry no
        # content. Reporting them as "malformed" would blame the backend for two outcomes
        # the protocol defines, and would hide the one the operator can actually fix.
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise JudgeRefusalError(f"The judge refused to answer: {refusal.strip()}")

        if first_choice.get("finish_reason") == "length":
            raise TruncatedResponseError(
                f"The judge stopped at the {max_tokens}-token budget before finishing its "
                "response, so the JSON is incomplete. Raise SLMEVAL_MAX_TOKENS."
            )

        text = message.get("content")
        if not isinstance(text, str):
            raise ValueError("Malformed chat-completions response body")

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
