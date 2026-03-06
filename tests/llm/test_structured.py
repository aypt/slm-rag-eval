from __future__ import annotations

import pytest
from pydantic import BaseModel

from slm_rag_eval.llm.errors import JSONGenerationError
from slm_rag_eval.llm.structured import generate_json
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
    assert fake_llm.calls[0].json_schema == JudgeOutput.model_json_schema()


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
