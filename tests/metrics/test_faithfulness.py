from __future__ import annotations

from pathlib import Path

import pytest

from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.metrics.faithfulness import score_faithfulness
from tests.conftest import FakeLLMClient

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _verdict(claim: str, verdict: str, reason: str = "The context gives this evidence.") -> str:
    return f'{{"claim":"{claim}","verdict":"{verdict}","reason":"{reason}"}}'


def _batch(*items: str) -> str:
    """Verdicts are object-wrapped: a top-level array is not a valid schema root."""
    return f'{{"verdicts":[{",".join(items)}]}}'


def _user_turn(call: object) -> str:
    """The user message of a recorded call, without the system instructions."""
    messages = call.messages  # type: ignore[attr-defined]  # RecordedCall in conftest
    return "\n".join(str(m["content"]) for m in messages if m["role"] == "user")


def _render_messages(messages: list[dict[str, object]]) -> str:
    return "\n\n".join(f"[{message['role']}]\n{message['content']}" for message in messages)


async def test_strict_scoring_counts_uncertain_as_unsupported(fake_llm: FakeLLMClient) -> None:
    fake_llm.push(
        '{"claims":["Claim A.","Claim B.","Claim C."]}',
        _batch(
            _verdict("Claim A.", "supported"),
            _verdict("Claim B.", "unsupported"),
            _verdict("Claim C.", "uncertain"),
        ),
    )

    result = await score_faithfulness(
        EvalRequest(question="Which claims hold?", answer="Three claims.", contexts=["Evidence."]),
        fake_llm,
    )

    assert result.faithfulness == pytest.approx(1 / 3)
    assert [item.verdict for item in result.verdicts] == [
        "supported",
        "unsupported",
        "uncertain",
    ]
    assert result.model_info == {
        "model": "fake",
        "k": 1,
        "strict": True,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    assert set(result.timings) == {"claim_extraction_ms", "verification_ms"}


async def test_non_strict_scoring_excludes_uncertain(fake_llm: FakeLLMClient) -> None:
    fake_llm.push(
        '{"claims":["Claim A.","Claim B.","Claim C."]}',
        _batch(
            _verdict("Claim A.", "supported"),
            _verdict("Claim B.", "unsupported"),
            _verdict("Claim C.", "uncertain"),
        ),
    )

    result = await score_faithfulness(
        EvalRequest(question="Question", answer="Answer", contexts=["Context"]),
        fake_llm,
        strict=False,
    )

    assert result.faithfulness == 0.5
    assert result.model_info["strict"] is False


async def test_non_strict_scoring_is_none_when_all_claims_are_uncertain(
    fake_llm: FakeLLMClient,
) -> None:
    fake_llm.push(
        '{"claims":["Claim A.","Claim B."]}',
        _batch(
            _verdict("Claim A.", "uncertain"),
            _verdict("Claim B.", "uncertain"),
        ),
    )

    result = await score_faithfulness(
        EvalRequest(question="Question", answer="Answer", contexts=[]),
        fake_llm,
        strict=False,
    )

    assert result.faithfulness is None


async def test_seven_claims_are_verified_in_batches_of_five(
    fake_llm: FakeLLMClient,
) -> None:
    claims = [f"Claim {index}." for index in range(7)]
    fake_llm.push(
        '{"claims":[' + ",".join(f'"{claim}"' for claim in claims) + "]}",
        _batch(*(_verdict(claim, "supported") for claim in claims[:5])),
        _batch(*(_verdict(claim, "supported") for claim in claims[5:])),
    )

    result = await score_faithfulness(
        EvalRequest(question="Question", answer="Seven facts.", contexts=["All evidence."]),
        fake_llm,
    )

    assert result.faithfulness == 1.0
    assert len(fake_llm.calls) == 3
    # Counted in the user turn only: the system instructions now show the output shape,
    # which contains a `"claim":` of its own and would inflate a whole-prompt count.
    assert _user_turn(fake_llm.calls[1]).count('"claim":') == 5
    assert _user_turn(fake_llm.calls[2]).count('"claim":') == 2


async def test_empty_answer_returns_explanation_without_calling_judge(
    fake_llm: FakeLLMClient,
) -> None:
    result = await score_faithfulness(
        EvalRequest(question="Question", answer="  \n", contexts=["Context"]),
        fake_llm,
    )

    assert result.faithfulness is None
    assert result.verdicts[0].verdict == "uncertain"
    assert "empty" in result.verdicts[0].reason.lower()
    assert fake_llm.calls == []


async def test_zero_claims_retries_extraction_once_then_returns_none(
    fake_llm: FakeLLMClient,
) -> None:
    fake_llm.push('{"claims":[]}', '{"claims":[]}')

    result = await score_faithfulness(
        EvalRequest(question="Question", answer="I prefer blue.", contexts=["Context"]),
        fake_llm,
    )

    assert result.faithfulness is None
    assert "two extraction attempts" in result.verdicts[0].reason
    assert len(fake_llm.calls) == 2
    assert "No factual claims were returned" in fake_llm.calls[1].full_text()


async def test_k_three_majority_vote_and_three_way_tie(fake_llm: FakeLLMClient) -> None:
    fake_llm.push(
        '{"claims":["Claim A.","Claim B."]}',
        _batch(_verdict("Claim A.", "supported"), _verdict("Claim B.", "supported")),
        _batch(_verdict("Claim A.", "supported"), _verdict("Claim B.", "unsupported")),
        _batch(_verdict("Claim A.", "unsupported"), _verdict("Claim B.", "uncertain")),
    )

    result = await score_faithfulness(
        EvalRequest(question="Question", answer="Two facts.", contexts=["Some evidence."]),
        fake_llm,
        k=3,
    )

    assert [item.verdict for item in result.verdicts] == ["supported", "uncertain"]
    assert result.faithfulness == 0.5
    assert result.model_info["k"] == 3
    assert len(fake_llm.calls) == 4


async def test_verdict_count_mismatch_is_reasked_once(fake_llm: FakeLLMClient) -> None:
    fake_llm.push(
        '{"claims":["Claim A.","Claim B."]}',
        _batch(_verdict("Claim A.", "supported")),
        _batch(_verdict("Claim A.", "supported"), _verdict("Claim B.", "unsupported")),
    )

    result = await score_faithfulness(
        EvalRequest(question="Question", answer="Two facts.", contexts=["Evidence."]),
        fake_llm,
    )

    assert result.faithfulness == 0.5
    assert "returned 1 items for 2 claims" in fake_llm.calls[2].full_text()


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
async def test_k_must_be_a_positive_integer(fake_llm: FakeLLMClient, k: object) -> None:
    with pytest.raises(ValueError, match="k must be an integer"):
        await score_faithfulness(
            EvalRequest(question="Question", answer="Answer", contexts=[]),
            fake_llm,
            k=k,  # type: ignore[arg-type] -- intentionally invalid input exercises validation.
        )


async def test_rendered_prompts_match_snapshots(fake_llm: FakeLLMClient) -> None:
    fake_llm.push(
        '{"claims":["Orion launched on 14 June 2031."]}',
        _batch(
            _verdict(
                "Orion launched on 14 June 2031.",
                "supported",
                "The context states the same launch date.",
            )
        ),
    )
    request = EvalRequest(
        question="When did Orion launch?",
        answer="Orion launched on 14 June 2031.",
        contexts=["The Orion mission launched on 14 June 2031."],
    )

    await score_faithfulness(request, fake_llm)

    extraction = _render_messages(fake_llm.calls[0].messages)
    verification = _render_messages(fake_llm.calls[1].messages)
    assert extraction == (SNAPSHOT_DIR / "claim_extraction.txt").read_text().rstrip("\n")
    assert verification == (SNAPSHOT_DIR / "claim_verification.txt").read_text().rstrip("\n")
    assert extraction.count("Example ") == 2
    assert len(extraction.split()) < 1200
