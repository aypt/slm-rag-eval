from __future__ import annotations

from slm_rag_eval.core.config import Settings
from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.metrics.registry import evaluate
from slm_rag_eval.privacy.sanitizer import Sanitizer
from tests.conftest import FakeLLMClient
from tests.privacy.test_sanitizer import StubAnalyzer


async def test_privacy_invariant_no_pii_leaves_process(fake_llm: FakeLLMClient) -> None:
    # SECURITY INVARIANT: this assertion must never be weakened (AGENTS.md rule 2).
    synthetic_name = "Rowan Quill"
    synthetic_email = "rowan@example.com"
    synthetic_phone = "202-555-0147"
    analyzer = StubAnalyzer(
        {
            synthetic_name: "PERSON",
            synthetic_email: "EMAIL_ADDRESS",
            synthetic_phone: "PHONE_NUMBER",
        }
    )
    sanitizer = Sanitizer(analyzer=analyzer)
    fake_llm.push(
        '{"claims":["<PERSON_1> uses <EMAIL_ADDRESS_1>."]}',
        '[{"claim":"<PERSON_1> uses <EMAIL_ADDRESS_1>.","verdict":"supported",'
        '"reason":"The context identifies <PERSON_1> and <PHONE_NUMBER_1>."}]',
        '{"questions":["Who uses <EMAIL_ADDRESS_1>?",'
        '"Which email does <PERSON_1> use?","Whose email is <EMAIL_ADDRESS_1>?"]}',
        '{"ratings":['
        '{"rating":2,"reason":"The intent matches."},'
        '{"rating":2,"reason":"The intent matches."},'
        '{"rating":1,"reason":"The question is phrased more broadly."}'
        "]}",
    )
    request = EvalRequest(
        question=f"What contact details belong to {synthetic_name}?",
        answer=f"{synthetic_name} uses {synthetic_email}.",
        contexts=[f"Call {synthetic_name} at {synthetic_phone}."],
    )

    result = await evaluate(
        request,
        judge=fake_llm,
        metrics=["faithfulness", "relevance"],
        settings=Settings(privacy_mode="mask", _env_file=None),
        sanitizer=sanitizer,
    )

    assert fake_llm.calls
    for call in fake_llm.calls:
        outbound = call.full_text()
        assert synthetic_name not in outbound
        assert synthetic_email not in outbound
        assert synthetic_phone not in outbound
    assert result.verdicts[0].claim == f"{synthetic_name} uses {synthetic_email}."
    assert synthetic_name in result.verdicts[0].reason
    assert synthetic_phone in result.verdicts[0].reason


class FailIfCalledSanitizer:
    def sanitize(self, texts: list[str]) -> None:
        raise AssertionError("privacy_mode='off' must bypass sanitization")


async def test_privacy_mode_off_bypasses_sanitizer(fake_llm: FakeLLMClient) -> None:
    fake_llm.push(
        '{"claims":["The synthetic module is shielded."]}',
        '[{"claim":"The synthetic module is shielded.","verdict":"supported",'
        '"reason":"The context supports the claim."}]',
    )
    request = EvalRequest(
        question="Is the synthetic module shielded?",
        answer="The synthetic module is shielded.",
        contexts=["The synthetic module is shielded."],
    )

    await evaluate(
        request,
        judge=fake_llm,
        settings=Settings(privacy_mode="off", _env_file=None),
        sanitizer=FailIfCalledSanitizer(),
    )

    assert len(fake_llm.calls) == 2
