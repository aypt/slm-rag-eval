"""Answer relevance scored through reverse question generation and comparison."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from slm_rag_eval.core.schemas import (
    DistinctNonBlankList,
    EvalRequest,
    EvalResult,
    NonBlankStr,
)
from slm_rag_eval.llm.client import LLMClient, LLMResponse
from slm_rag_eval.llm.structured import generate_json

_QUESTION_COUNT = 3

_GENERATION_INSTRUCTIONS = """Generate exactly three distinct questions that the answer
directly answers.

Rules:
- Use only information stated in the answer.
- Each question must be concise, self-contained, and directly answerable by the answer.
- Do not introduce facts, entities, or assumptions that are absent from the answer.
- Return only JSON matching {"questions": ["question", "question", "question"]}.
"""

_RATING_INSTRUCTIONS = """Rate how well each generated question matches the original question.

Use this scale:
- 2: the questions have the same intent and seek the same information.
- 1: the questions overlap but differ in an important part of their intent or requested information.
- 0: the questions are unrelated or seek different information.

Rules:
- Return exactly one rating for each generated question, in the same index order.
- Use only ratings 0, 1, or 2.
- Give a concise, one-sentence reason for each rating.
- Return only JSON matching the required schema.
"""


class GeneratedQuestions(BaseModel):
    """The three questions inferred from an answer.

    Counting alone is not enough: three blank strings, or the same question three times,
    satisfy a length check and then average to a perfect relevance score.
    """

    questions: DistinctNonBlankList = Field(
        min_length=_QUESTION_COUNT, max_length=_QUESTION_COUNT
    )


class QuestionRating(BaseModel):
    """Similarity judgment for one generated question."""

    rating: Literal[0, 1, 2]
    reason: NonBlankStr


class QuestionRatings(BaseModel):
    """Index-aligned ratings for all generated questions."""

    ratings: list[QuestionRating] = Field(
        min_length=_QUESTION_COUNT,
        max_length=_QUESTION_COUNT,
    )


class _RecordingClient:
    """Transparent client wrapper that accumulates completion metadata."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = "unknown"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Delegate to the wrapped client, accumulating token usage and model name."""
        response = await self._client.complete(
            messages,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt_tokens += response.usage.get("prompt_tokens", 0)
        self.completion_tokens += response.usage.get("completion_tokens", 0)
        self.model = response.model
        return response


def _model_info(client: _RecordingClient) -> dict[str, object]:
    return {
        "model": client.model,
        "token_usage": {
            "prompt_tokens": client.prompt_tokens,
            "completion_tokens": client.completion_tokens,
            "total_tokens": client.prompt_tokens + client.completion_tokens,
        },
    }


def _generation_messages(answer: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": _GENERATION_INSTRUCTIONS},
        {
            "role": "user",
            "content": f"Answer:\n{json.dumps(answer, ensure_ascii=False)}",
        },
    ]


def _rating_messages(
    original_question: str,
    generated_questions: list[str],
) -> list[dict[str, Any]]:
    generated_payload = [
        {"index": index, "question": question}
        for index, question in enumerate(generated_questions)
    ]
    content = (
        "Original question:\n"
        f"{json.dumps(original_question, ensure_ascii=False)}\n\n"
        "Generated questions:\n"
        f"{json.dumps(generated_payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": _RATING_INSTRUCTIONS},
        {"role": "user", "content": content},
    ]


async def score_relevance(request: EvalRequest, judge: LLMClient) -> EvalResult:
    """Score whether the answer addresses the original question."""
    client = _RecordingClient(judge)
    if not request.answer.strip():
        return EvalResult(
            relevance=None,
            model_info=_model_info(client),
            timings={"question_generation_ms": 0.0, "question_rating_ms": 0.0},
        )

    generation_started = perf_counter()
    generated = await generate_json(
        client,
        _generation_messages(request.answer),
        GeneratedQuestions,
    )
    generation_ms = (perf_counter() - generation_started) * 1000

    rating_started = perf_counter()
    rated = await generate_json(
        client,
        _rating_messages(request.question, generated.questions),
        QuestionRatings,
    )
    rating_ms = (perf_counter() - rating_started) * 1000
    relevance = sum(item.rating / 2 for item in rated.ratings) / len(rated.ratings)

    return EvalResult(
        relevance=relevance,
        model_info=_model_info(client),
        timings={
            "question_generation_ms": generation_ms,
            "question_rating_ms": rating_ms,
        },
    )
