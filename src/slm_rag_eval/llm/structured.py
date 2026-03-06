"""Schema-validated JSON generation with a bounded repair loop."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from slm_rag_eval.llm.client import LLMClient
from slm_rag_eval.llm.errors import JSONGenerationError

ModelT = TypeVar("ModelT", bound=BaseModel)
_MAX_ATTEMPTS = 3


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].strip().lower() in {"```", "```json"}:
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


async def generate_json(
    client: LLMClient,
    messages: list[dict[str, Any]],
    schema: type[ModelT],
) -> ModelT:
    """Generate and validate JSON, asking the model to repair invalid responses."""
    conversation = [dict(message) for message in messages]
    json_schema = schema.model_json_schema()
    last_raw_text = ""
    last_error = "No response was generated"

    for attempt_number in range(1, _MAX_ATTEMPTS + 1):
        response = await client.complete(conversation, json_schema=json_schema)
        last_raw_text = response.text
        try:
            return schema.model_validate_json(_strip_markdown_fence(last_raw_text))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            if attempt_number < _MAX_ATTEMPTS:
                conversation.extend(
                    [
                        {"role": "assistant", "content": last_raw_text},
                        {
                            "role": "user",
                            "content": (
                                "Your response was not valid JSON for the required schema. "
                                f"Validation error: {last_error}. Return only corrected JSON."
                            ),
                        },
                    ]
                )

    raise JSONGenerationError(
        f"Failed to generate schema-valid JSON after {_MAX_ATTEMPTS} attempts: {last_error}",
        last_raw_text=last_raw_text,
    )
