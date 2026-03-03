from __future__ import annotations

import pytest

from tests.conftest import FakeLLMClient


async def test_fake_llm_returns_in_order_and_records_calls() -> None:
    fake = FakeLLMClient()
    fake.push("first", "second")

    r1 = await fake.complete([{"role": "user", "content": "hello"}], json_schema={"type": "object"})
    r2 = await fake.complete([{"role": "user", "content": "world"}])

    assert (r1.text, r2.text) == ("first", "second")
    assert len(fake.calls) == 2
    assert fake.calls[0].json_schema == {"type": "object"}
    assert "hello" in fake.calls[0].full_text()


async def test_fake_llm_raises_when_out_of_responses() -> None:
    fake = FakeLLMClient()
    with pytest.raises(AssertionError):
        await fake.complete([{"role": "user", "content": "x"}])
