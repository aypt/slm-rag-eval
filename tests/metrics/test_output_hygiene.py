"""A schema-valid judge response can still be semantically empty — these pin the rejection.

Blank and repeated outputs are the two ways a weak judge corrupts a score without ever
producing invalid JSON: a blank claim marked `supported` averages to a perfect 1.0, and a
claim returned twice carries twice the weight of every other claim. Both used to score
silently. They must now go through the repair loop and, if the judge will not fix them,
fail loudly rather than quietly.
"""

from __future__ import annotations

import json

import pytest

from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.llm.client import LLMResponse
from slm_rag_eval.llm.errors import JSONGenerationError
from slm_rag_eval.metrics.faithfulness import score_faithfulness
from slm_rag_eval.metrics.relevance import score_relevance
from tests.support import verdict_batch

CLAIM = "The probe carried three instruments."
CONTEXT = "The probe carried three instruments."


class StubbornJudge:
    """Returns one fixed reply however many times it is asked to repair it."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def complete(self, messages: list[dict[str, object]], **kwargs: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.reply, usage={}, model="stub")


def _request() -> EvalRequest:
    return EvalRequest(question="How many instruments?", answer="Three.", contexts=[CONTEXT])


@pytest.mark.parametrize(
    ("name", "claims"),
    [
        ("blank", ["   "]),
        ("empty string", [""]),
        ("duplicate", [CLAIM, CLAIM]),
        ("duplicate ignoring case and spacing", [CLAIM, f"  {CLAIM.upper()}  "]),
        ("one real claim plus a blank", [CLAIM, "  "]),
    ],
)
async def test_unusable_claim_lists_never_produce_a_score(name: str, claims: list[str]) -> None:
    judge = StubbornJudge(json.dumps({"claims": claims}))

    with pytest.raises(JSONGenerationError):
        await score_faithfulness(_request(), judge)

    # The repair loop was actually used rather than the response being taken at face value.
    assert judge.calls > 1


async def test_a_blank_claim_would_otherwise_have_scored_a_perfect_one() -> None:
    """The specific regression: `claims=["   "]` marked supported used to give 1.0."""
    judge = StubbornJudge(json.dumps({"claims": ["   "]}))

    with pytest.raises(JSONGenerationError, match="blank"):
        await score_faithfulness(_request(), judge)


async def test_a_repeated_claim_is_named_in_the_repair_message() -> None:
    judge = StubbornJudge(json.dumps({"claims": [CLAIM, CLAIM]}))

    with pytest.raises(JSONGenerationError, match="repeat"):
        await score_faithfulness(_request(), judge)


async def test_claims_keep_their_wording_and_are_only_stripped() -> None:
    """Validation must normalize whitespace only — never rewrite what the judge said."""
    padded = f"  {CLAIM}  "
    judge = _scripted(
        json.dumps({"claims": [padded]}),
        verdict_batch(CLAIM),
    )

    result = await score_faithfulness(_request(), judge)

    assert result.faithfulness == 1.0
    assert result.verdicts[0].claim == CLAIM


@pytest.mark.parametrize(
    ("name", "questions"),
    [
        ("all blank", ["", "  ", ""]),
        ("all identical", ["Same question?"] * 3),
        ("two identical", ["A?", "B?", "b?  "]),
    ],
)
async def test_unusable_question_lists_never_produce_a_relevance_score(
    name: str, questions: list[str]
) -> None:
    judge = StubbornJudge(json.dumps({"questions": questions}))

    with pytest.raises(JSONGenerationError):
        await score_relevance(_request(), judge)


async def test_a_blank_verdict_reason_is_rejected() -> None:
    """A verdict with no reason is unauditable: the report cannot show why a claim failed."""
    judge = _scripted(
        json.dumps({"claims": [CLAIM]}),
        *[verdict_batch(CLAIM, reason="   ")] * 3,
    )

    with pytest.raises(JSONGenerationError):
        await score_faithfulness(_request(), judge)


def _scripted(*replies: str) -> StubbornJudge:
    """A judge that walks through `replies` in order, then repeats the last one forever."""

    class Scripted(StubbornJudge):
        def __init__(self) -> None:
            super().__init__(replies[-1])
            self.queue = list(replies)

        async def complete(
            self, messages: list[dict[str, object]], **kwargs: object
        ) -> LLMResponse:
            self.calls += 1
            text = self.queue.pop(0) if self.queue else self.reply
            return LLMResponse(text=text, usage={}, model="stub")

    return Scripted()
