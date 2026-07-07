"""Hybrid-reasoning judges must answer directly, or the token budget is spent on thinking.

Measured against qwen3-8b through an OpenRouter route: 339 completion tokens and 5.6 s per
call by default, 47 tokens and 1.7 s with the directive, and the same three claims either
way. Over a 150-sample run that difference is hours of rented GPU time.

The directive is applied at the prompt rather than as a request parameter because
`reasoning.enabled` is a gateway field a local Ollama does not implement, while the chat
template handles this everywhere the model is served.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm.client import (
    NO_THINK_DIRECTIVE,
    OpenAICompatClient,
    apply_thinking_directive,
)


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You extract claims."},
        {"role": "user", "content": "Extract from: The probe carried three instruments."},
    ]


def test_the_directive_lands_on_the_final_user_turn() -> None:
    updated = apply_thinking_directive(_messages())

    assert updated[-1]["content"].endswith(NO_THINK_DIRECTIVE)
    assert updated[0]["content"] == "You extract claims.", "the system prompt is untouched"


def test_the_input_messages_are_never_mutated() -> None:
    """The caller keeps its list for a repair round; mutating it would compound directives."""
    original = _messages()
    snapshot = json.dumps(original)

    apply_thinking_directive(original)

    assert json.dumps(original) == snapshot


def test_an_already_directed_prompt_is_left_alone() -> None:
    once = apply_thinking_directive(_messages())

    assert apply_thinking_directive(once) == once


def test_only_the_last_user_turn_is_directed_in_a_repair_conversation() -> None:
    """A repair round appends turns; the directive belongs on the newest one only."""
    conversation = [
        *_messages(),
        {"role": "assistant", "content": "{}"},
        {"role": "user", "content": "That was not valid JSON. Try again."},
    ]

    updated = apply_thinking_directive(conversation)

    assert updated[1]["content"].count(NO_THINK_DIRECTIVE) == 0
    assert updated[3]["content"].endswith(NO_THINK_DIRECTIVE)


def test_messages_without_a_user_turn_are_returned_unchanged() -> None:
    system_only = [{"role": "system", "content": "instructions"}]

    assert apply_thinking_directive(system_only) == system_only


def _client_capturing(settings: Settings) -> tuple[OpenAICompatClient, list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "model": "stub",
            },
        )

    client = OpenAICompatClient(
        settings, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return client, captured


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {"base_url": "http://judge.test/v1", "model": "stub"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.mark.parametrize("disabled", [True, False])
async def test_the_setting_controls_what_goes_on_the_wire(disabled: bool) -> None:
    client, captured = _client_capturing(_settings(disable_thinking=disabled))

    await client.complete(_messages())

    sent = captured[0]["messages"][-1]["content"]
    assert (NO_THINK_DIRECTIVE in sent) is disabled
