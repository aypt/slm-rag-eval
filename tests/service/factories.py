"""Helpers for driving the service in tests without a network, a model, or a real DB server.

Kept out of tests/conftest.py deliberately: AGENTS.md forbids changing existing fixtures.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm.client import LLMResponse
from slm_rag_eval.service.api import create_app
from tests.conftest import FakeLLMClient


class RaisingLLMClient:
    """Judge that always fails, counting attempts so retry behavior can be asserted."""

    def __init__(self, message: str = "judge backend is down") -> None:
        self.message = message
        self.attempts = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.attempts += 1
        raise RuntimeError(self.message)


def service_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings pointing at a throwaway SQLite file, with the embedded worker on."""
    defaults: dict[str, Any] = {
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}",
        "worker_embedded": True,
        "worker_concurrency": 2,
        "privacy_mode": "off",
        "enabled_metrics": ["faithfulness"],
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def build_app(settings: Settings, judge: Any, **kwargs: Any) -> FastAPI:
    """App wired to a scripted judge; `kwargs` reach `create_app` (e.g. `sanitizer=`)."""
    return create_app(
        settings=settings,
        judge_factory=lambda: judge,
        worker_poll_interval_s=0.01,
        **kwargs,
    )


@contextlib.asynccontextmanager
async def running_app(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Run the app's lifespan (which starts the embedded worker) around an ASGI client."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def wait_for_settled(
    client: httpx.AsyncClient,
    job_id: str,
    *,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Poll one job until it is done or error; fail loudly rather than hang."""
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = await client.get(f"/v1/evaluations/{job_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] in {"done", "error"}:
            return last
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} never settled; last seen: {last}")


def faithfulness_script(claim: str) -> FakeLLMClient:
    """A judge scripted for one supported claim."""
    judge = FakeLLMClient()
    judge.push(
        '{"claims":["' + claim + '"]}',
        '{"verdicts":[{"claim":"' + claim + '","verdict":"supported",'
        '"reason":"The context states the same thing."}]}',
    )
    return judge
