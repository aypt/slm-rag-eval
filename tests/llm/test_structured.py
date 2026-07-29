from __future__ import annotations

import pytest
from pydantic import BaseModel

from slm_rag_eval.llm.errors import JSONGenerationError
from slm_rag_eval.llm.structured import closed_json_schema, generate_json
from tests.conftest import FakeLLMClient


class JudgeOutput(BaseModel):
    verdict: str
    confidence: float


async def test_generate_json_happy_path(fake_llm: FakeLLMClient) -> None:
    fake_llm.push('{"verdict": "supported", "confidence": 0.9}')

    result = await generate_json(
        fake_llm,
        [{"role": "user", "content": "Judge this answer."}],
        JudgeOutput,
    )

    assert result == JudgeOutput(verdict="supported", confidence=0.9)
    # The schema is sent closed: OpenAI rejects an object without
    # `additionalProperties: false`, and this assertion used to pin the raw form.
    assert fake_llm.calls[0].json_schema == closed_json_schema(JudgeOutput)
    assert fake_llm.calls[0].json_schema["additionalProperties"] is False


async def test_generate_json_strips_markdown_fence(fake_llm: FakeLLMClient) -> None:
    fake_llm.push('```json\n{"verdict": "uncertain", "confidence": 0.4}\n```')

    result = await generate_json(
        fake_llm,
        [{"role": "user", "content": "Judge this answer."}],
        JudgeOutput,
    )

    assert result.verdict == "uncertain"
    assert result.confidence == 0.4


async def test_generate_json_repairs_invalid_first_response(fake_llm: FakeLLMClient) -> None:
    fake_llm.push(
        '{"verdict": "supported"}',
        '{"verdict": "supported", "confidence": 0.8}',
    )
    original_messages = [{"role": "user", "content": "Judge this answer."}]

    result = await generate_json(fake_llm, original_messages, JudgeOutput)

    assert result.confidence == 0.8
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[1].messages[-1]["role"] == "user"
    assert "Validation error" in fake_llm.calls[1].messages[-1]["content"]
    assert fake_llm.calls[1].messages[-2] == {
        "role": "assistant",
        "content": '{"verdict": "supported"}',
    }
    assert original_messages == [{"role": "user", "content": "Judge this answer."}]


async def test_generate_json_raises_after_three_failures(fake_llm: FakeLLMClient) -> None:
    fake_llm.push("not json", "still not json", "last invalid response")

    with pytest.raises(JSONGenerationError) as exc_info:
        await generate_json(
            fake_llm,
            [{"role": "user", "content": "Judge this answer."}],
            JudgeOutput,
        )

    assert len(fake_llm.calls) == 3
    assert exc_info.value.last_raw_text == "last invalid response"
    assert "after 3 attempts" in str(exc_info.value)


def test_nested_objects_are_closed_including_through_defs() -> None:
    """OpenAI's validator walks `$defs` too, and rejected our verdict schema there."""
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str

    class Wrapper(BaseModel):
        items: list[Item]

    schema = closed_json_schema(Wrapper)

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Item"]["additionalProperties"] is False


def test_an_explicit_additional_properties_is_not_overwritten() -> None:
    """Closing is a default, not a policy: a schema that opts in keeps its choice."""
    from slm_rag_eval.llm.structured import _close_objects

    closed = _close_objects({"type": "object", "additionalProperties": True})

    assert closed["additionalProperties"] is True


def test_non_object_nodes_are_left_alone() -> None:
    from slm_rag_eval.llm.structured import _close_objects

    assert _close_objects({"type": "array", "items": {"type": "string"}}) == {
        "type": "array",
        "items": {"type": "string"},
    }
