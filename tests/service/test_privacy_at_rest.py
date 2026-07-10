from __future__ import annotations

import json
from pathlib import Path

from slm_rag_eval.privacy.sanitizer import Sanitizer
from slm_rag_eval.service.db import create_engine, create_session_factory, create_tables, get_job
from tests.conftest import FakeLLMClient
from tests.privacy.test_sanitizer import StubAnalyzer
from tests.service.factories import (
    build_app,
    running_app,
    service_settings,
    wait_for_settled,
)

# Invented data only — never real personal data (AGENTS.md rule 3).
NAME = "Rowan Quill"
EMAIL = "rowan@example.com"
PHONE = "202-555-0147"

_SUBMISSION = {
    "question": f"What contact details belong to {NAME}?",
    "answer": f"{NAME} uses {EMAIL}.",
    "contexts": [f"Call {NAME} at {PHONE}."],
}


def _stub_sanitizer() -> Sanitizer:
    return Sanitizer(
        analyzer=StubAnalyzer(
            {NAME: "PERSON", EMAIL: "EMAIL_ADDRESS", PHONE: "PHONE_NUMBER"},
        )
    )


def _masked_judge() -> FakeLLMClient:
    judge = FakeLLMClient()
    judge.push(
        '{"claims":["<PERSON_1> uses <EMAIL_ADDRESS_1>."]}',
        '{"verdicts":[{"claim":"<PERSON_1> uses <EMAIL_ADDRESS_1>.","verdict":"supported",'
        '"reason":"The context lists <PERSON_1> and <PHONE_NUMBER_1>."}]}',
    )
    return judge


async def test_privacy_invariant_no_pii_is_persisted_or_sent(tmp_path: Path) -> None:
    """SECURITY INVARIANT: this assertion must never be weakened (AGENTS.md rule 2).

    Masking happens at the API boundary, so the whole downstream path — the job row, the
    judge prompts, and the stored result — must be free of the raw values.
    """
    judge = _masked_judge()
    settings = service_settings(tmp_path, privacy_mode="mask")
    engine = create_engine(settings.database_url)
    try:
        await create_tables(engine)
        session_factory = create_session_factory(engine)
        app = build_app(settings, judge, engine=engine, sanitizer=_stub_sanitizer())

        async with running_app(app) as client:
            submitted = await client.post("/v1/evaluations", json=_SUBMISSION)
            job_id = submitted.json()["job_id"]
            settled = await wait_for_settled(client, job_id)

            stored = await get_job(session_factory, job_id)
    finally:
        await engine.dispose()

    assert settled["status"] == "done"
    assert stored is not None

    persisted = json.dumps(stored.request_json)
    outbound = "\n".join(call.full_text() for call in judge.calls)
    served = json.dumps(settled)
    assert judge.calls, "the judge was never called, so the invariant would be vacuous"
    for secret in (NAME, EMAIL, PHONE):
        assert secret not in persisted
        assert secret not in outbound
        assert secret not in served

    assert "<PERSON_1>" in persisted
    assert "<EMAIL_ADDRESS_1>" in persisted
    assert "<PHONE_NUMBER_1>" in persisted


async def test_privacy_mode_off_persists_the_request_unchanged(tmp_path: Path) -> None:
    judge = FakeLLMClient()
    judge.push(
        '{"claims":["' + NAME + ' uses ' + EMAIL + '."]}',
        '{"verdicts":[{"claim":"' + NAME + ' uses ' + EMAIL + '.","verdict":"supported",'
        '"reason":"The context lists the same contact."}]}',
    )
    settings = service_settings(tmp_path, privacy_mode="off")
    engine = create_engine(settings.database_url)
    try:
        await create_tables(engine)
        session_factory = create_session_factory(engine)
        app = build_app(settings, judge, engine=engine, sanitizer=_stub_sanitizer())

        async with running_app(app) as client:
            submitted = await client.post(
                "/v1/evaluations", json={**_SUBMISSION, "options": {"privacy_mode": "off"}}
            )
            job_id = submitted.json()["job_id"]
            await wait_for_settled(client, job_id)
            stored = await get_job(session_factory, job_id)
    finally:
        await engine.dispose()

    assert stored is not None
    # `off` is an explicit, documented opt-out: what was submitted is what is stored.
    assert stored.request_json["request"]["answer"] == f"{NAME} uses {EMAIL}."
    assert stored.request_json["options"]["privacy_mode"] == "off"
