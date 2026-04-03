from __future__ import annotations

from pathlib import Path

from slm_rag_eval.service.db import (
    claim_next_queued_job,
    create_engine,
    create_session_factory,
    create_tables,
    get_job,
)
from tests.service.factories import (
    RaisingLLMClient,
    build_app,
    faithfulness_script,
    running_app,
    service_settings,
    wait_for_settled,
)

_SUBMISSION = {
    "question": "Which material shields the module?",
    "answer": "The module uses a ceramic shield.",
    "contexts": ["The module is protected by a ceramic shield."],
}


async def test_submit_then_embedded_worker_completes_the_job(tmp_path: Path) -> None:
    judge = faithfulness_script("The module uses a ceramic shield.")
    settings = service_settings(tmp_path)

    async with running_app(build_app(settings, judge)) as client:
        submitted = await client.post("/v1/evaluations", json=_SUBMISSION)
        assert submitted.status_code == 202
        assert submitted.json()["status"] == "queued"
        job_id = submitted.json()["job_id"]

        settled = await wait_for_settled(client, job_id)

    assert settled["status"] == "done"
    assert settled["error"] is None
    assert settled["result"]["faithfulness"] == 1.0
    assert [verdict["verdict"] for verdict in settled["result"]["verdicts"]] == ["supported"]
    assert len(judge.calls) == 2


async def test_submitted_options_override_settings_defaults(tmp_path: Path) -> None:
    judge = faithfulness_script("The module uses a ceramic shield.")
    judge.push(
        '{"questions":["Which material shields the module?",'
        '"What shield does the module use?","What protects the module?"]}',
        '{"ratings":['
        '{"rating":2,"reason":"Same intent."},'
        '{"rating":2,"reason":"Same intent."},'
        '{"rating":1,"reason":"Broader question."}'
        "]}",
    )
    settings = service_settings(tmp_path)

    async with running_app(build_app(settings, judge)) as client:
        submitted = await client.post(
            "/v1/evaluations",
            json={**_SUBMISSION, "options": {"metrics": ["faithfulness", "relevance"]}},
        )
        settled = await wait_for_settled(client, submitted.json()["job_id"])

    assert settled["status"] == "done"
    assert settled["result"]["faithfulness"] == 1.0
    assert settled["result"]["relevance"] is not None
    assert set(settled["result"]["model_info"]) == {"faithfulness", "relevance"}


async def test_unknown_job_returns_404(tmp_path: Path) -> None:
    judge = faithfulness_script("Unused.")
    settings = service_settings(tmp_path, worker_embedded=False)

    async with running_app(build_app(settings, judge)) as client:
        response = await client.get("/v1/evaluations/does-not-exist")

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


async def test_failing_judge_marks_the_job_error_after_one_retry(tmp_path: Path) -> None:
    judge = RaisingLLMClient("judge backend is down")
    settings = service_settings(tmp_path)

    async with running_app(build_app(settings, judge)) as client:
        submitted = await client.post("/v1/evaluations", json=_SUBMISSION)
        settled = await wait_for_settled(client, submitted.json()["job_id"])

    assert settled["status"] == "error"
    assert settled["result"] is None
    assert "judge backend is down" in settled["error"]
    # One initial attempt plus exactly one retry, each of which calls the judge once.
    assert judge.attempts == 2


async def test_queued_job_is_claimed_exactly_once(tmp_path: Path) -> None:
    """The conditional UPDATE is the claim, so a second claim of the same row finds nothing."""
    judge = faithfulness_script("Unused.")
    settings = service_settings(tmp_path, worker_embedded=False)
    engine = create_engine(settings.database_url)
    try:
        await create_tables(engine)
        session_factory = create_session_factory(engine)

        async with running_app(build_app(settings, judge, engine=engine)) as client:
            submitted = await client.post("/v1/evaluations", json=_SUBMISSION)
            job_id = submitted.json()["job_id"]

            first = await claim_next_queued_job(session_factory)
            second = await claim_next_queued_job(session_factory)

            assert first is not None
            assert first.id == job_id
            assert second is None

            stored = await get_job(session_factory, job_id)
            assert stored is not None
            assert stored.status == "running"
    finally:
        await engine.dispose()
