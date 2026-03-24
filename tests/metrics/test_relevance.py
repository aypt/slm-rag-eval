from __future__ import annotations

import pytest

from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.metrics.relevance import score_relevance
from tests.conftest import FakeLLMClient


async def test_relevance_generates_three_questions_then_rates_them(
    fake_llm: FakeLLMClient,
) -> None:
    fake_llm.push(
        '{"questions":["What powers the probe?","What is the energy source?",'
        '"How does the probe get power?"]}',
        '{"ratings":['
        '{"rating":2,"reason":"Both questions ask what powers the probe."},'
        '{"rating":1,"reason":"The questions overlap but use a broader framing."},'
        '{"rating":0,"reason":"The original asks about a different probe detail."}'
        "]}",
    )
    request = EvalRequest(
        question="What powers the probe?",
        answer="The probe uses a compact solar array.",
        contexts=[],
    )

    result = await score_relevance(request, fake_llm)

    assert result.relevance == 0.5
    assert result.faithfulness is None
    assert result.model_info == {
        "model": "fake",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    assert set(result.timings) == {"question_generation_ms", "question_rating_ms"}
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[0].json_schema["title"] == "GeneratedQuestions"
    assert fake_llm.calls[1].json_schema["title"] == "QuestionRatings"
    assert "The probe uses a compact solar array." in fake_llm.calls[0].full_text()
    assert "What powers the probe?" in fake_llm.calls[1].full_text()
    assert "What is the energy source?" in fake_llm.calls[1].full_text()


async def test_empty_answer_returns_none_without_calling_judge(
    fake_llm: FakeLLMClient,
) -> None:
    result = await score_relevance(
        EvalRequest(question="What happened?", answer=" \n ", contexts=[]),
        fake_llm,
    )

    assert result.relevance is None
    assert result.timings == {"question_generation_ms": 0.0, "question_rating_ms": 0.0}
    assert fake_llm.calls == []


async def test_question_generation_requires_exactly_three_questions(
    fake_llm: FakeLLMClient,
) -> None:
    fake_llm.push(
        '{"questions":["Question one?","Question two?"]}',
        '{"questions":["Question one?","Question two?","Question three?"]}',
        '{"ratings":['
        '{"rating":2,"reason":"The intent is the same."},'
        '{"rating":2,"reason":"The intent is the same."},'
        '{"rating":2,"reason":"The intent is the same."}'
        "]}",
    )

    result = await score_relevance(
        EvalRequest(question="Question one?", answer="A direct answer.", contexts=[]),
        fake_llm,
    )

    assert result.relevance == pytest.approx(1.0)
    assert len(fake_llm.calls) == 3
    assert "Validation error" in fake_llm.calls[1].full_text()
