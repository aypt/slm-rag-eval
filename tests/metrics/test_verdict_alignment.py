"""Verdict batches must be aligned by content, not merely counted.

A judge that answers in a different order returns the right number of items, so a count
check alone lets every verdict attach to the wrong claim — silently producing a wrong
evaluation. These tests pin the content check that prevents that.
"""

from __future__ import annotations

import json

import pytest

from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.metrics.faithfulness import score_faithfulness
from tests.conftest import FakeLLMClient

CLAIM_A = "The probe carried three instruments."
CLAIM_B = "The probe launched in 1998."
CONTEXT = "The probe carried three instruments and launched in 2019."


def _request() -> EvalRequest:
    return EvalRequest(question="What is on record?", answer="Two facts.", contexts=[CONTEXT])


def _verdict(claim: str, verdict: str) -> dict[str, str]:
    return {"claim": claim, "verdict": verdict, "reason": "Reason grounded in the context."}


def _batch(*verdicts: dict[str, str]) -> str:
    """Object-wrapped, matching the schema the judge is actually given."""
    return json.dumps({"verdicts": list(verdicts)})


async def test_out_of_order_verdicts_are_reasked_and_then_aligned(
    fake_llm: FakeLLMClient,
) -> None:
    fake_llm.push(
        json.dumps({"claims": [CLAIM_A, CLAIM_B]}),
        # Same count, reversed order: the failure mode a count check cannot see.
        _batch(_verdict(CLAIM_B, "unsupported"), _verdict(CLAIM_A, "supported")),
        _batch(_verdict(CLAIM_A, "supported"), _verdict(CLAIM_B, "unsupported")),
    )

    result = await score_faithfulness(_request(), fake_llm)

    assert [(v.claim, v.verdict) for v in result.verdicts] == [
        (CLAIM_A, "supported"),
        (CLAIM_B, "unsupported"),
    ]
    assert result.faithfulness == 0.5
    # The re-ask names the offending item so the judge can correct it.
    assert "Item 0 carries claim" in fake_llm.calls[2].full_text()
    assert "copying each claim verbatim" in fake_llm.calls[2].full_text()


async def test_persistently_misordered_verdicts_raise_instead_of_misattributing(
    fake_llm: FakeLLMClient,
) -> None:
    reversed_batch = _batch(_verdict(CLAIM_B, "unsupported"), _verdict(CLAIM_A, "supported"))
    fake_llm.push(
        json.dumps({"claims": [CLAIM_A, CLAIM_B]}),
        reversed_batch,
        reversed_batch,
    )

    with pytest.raises(ValueError, match="stayed misaligned after one retry"):
        await score_faithfulness(_request(), fake_llm)


async def test_duplicated_claim_in_a_verdict_batch_is_rejected(
    fake_llm: FakeLLMClient,
) -> None:
    duplicated = _batch(_verdict(CLAIM_A, "supported"), _verdict(CLAIM_A, "supported"))
    fake_llm.push(
        json.dumps({"claims": [CLAIM_A, CLAIM_B]}),
        duplicated,
        duplicated,
    )

    with pytest.raises(ValueError, match="stayed misaligned after one retry"):
        await score_faithfulness(_request(), fake_llm)


async def test_substituted_claim_of_the_same_length_is_rejected(
    fake_llm: FakeLLMClient,
) -> None:
    """One claim silently replaced by something the answer never said, count unchanged."""
    substituted = _batch(
        _verdict(CLAIM_A, "supported"),
        _verdict("The probe was built by a private contractor.", "supported"),
    )
    fake_llm.push(
        json.dumps({"claims": [CLAIM_A, CLAIM_B]}),
        substituted,
        substituted,
    )

    with pytest.raises(ValueError, match="stayed misaligned after one retry"):
        await score_faithfulness(_request(), fake_llm)


async def test_whitespace_and_case_differences_still_count_as_aligned(
    fake_llm: FakeLLMClient,
) -> None:
    """Re-wrapped or re-cased echoes are the same claim; wording changes are not."""
    fake_llm.push(
        json.dumps({"claims": [CLAIM_A]}),
        _batch(_verdict("  the probe   carried three\ninstruments. ", "supported")),
    )

    result = await score_faithfulness(_request(), fake_llm)

    assert result.faithfulness == 1.0
    # The stored claim is the extracted one, not the judge's re-spacing.
    assert result.verdicts[0].claim == CLAIM_A
    assert len(fake_llm.calls) == 2  # no retry was needed
