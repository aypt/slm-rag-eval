from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from slm_rag_eval.service.logging import JsonFormatter, job_id_var, request_id_var
from tests.service.factories import build_app, faithfulness_script, running_app, service_settings


def _record(message: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("test.logger", logging.INFO, __file__, 10, message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_one_object_per_line() -> None:
    payload = json.loads(JsonFormatter().format(_record(status_code=200)))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "hello"
    assert payload["status_code"] == 200
    assert "timestamp" in payload
    # No correlation id is set in this context, so none is invented.
    assert "request_id" not in payload
    assert "job_id" not in payload


def test_json_formatter_includes_the_active_correlation_ids() -> None:
    request_token = request_id_var.set("req-1")
    job_token = job_id_var.set("job-1")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        request_id_var.reset(request_token)
        job_id_var.reset(job_token)

    assert payload["request_id"] == "req-1"
    assert payload["job_id"] == "job-1"


def test_json_formatter_serializes_exceptions() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = _record("failed")
        record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))

    assert "RuntimeError: boom" in payload["exception"]


async def test_every_response_carries_a_request_id(tmp_path: Path) -> None:
    settings = service_settings(tmp_path, worker_embedded=False)

    async with running_app(build_app(settings, faithfulness_script("unused"))) as client:
        generated = await client.get("/healthz")
        supplied = await client.get("/healthz", headers={"x-request-id": "caller-supplied"})

    assert generated.headers["x-request-id"]
    # A caller-supplied id is preserved so traces join up across services.
    assert supplied.headers["x-request-id"] == "caller-supplied"


async def test_unknown_metric_is_mapped_to_400_with_the_request_id(tmp_path: Path) -> None:
    settings = service_settings(tmp_path, worker_embedded=False)

    async with running_app(build_app(settings, faithfulness_script("unused"))) as client:
        response = await client.post(
            "/v1/evaluations",
            json={
                "question": "Q?",
                "answer": "A.",
                "contexts": [],
                "options": {"metrics": ["fluency"]},
            },
        )

    assert response.status_code == 400
    body = response.json()
    assert "fluency" in body["detail"]
    assert body["request_id"]


@pytest.mark.parametrize("bad_body", [{"answer": "A."}, {"question": "Q?", "answer": 5}])
async def test_malformed_bodies_are_rejected_before_any_work(
    tmp_path: Path, bad_body: dict[str, object]
) -> None:
    settings = service_settings(tmp_path, worker_embedded=False)
    judge = faithfulness_script("unused")

    async with running_app(build_app(settings, judge)) as client:
        response = await client.post("/v1/evaluations", json=bad_body)

    assert response.status_code == 422
    assert judge.calls == []
