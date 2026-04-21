"""Async evaluation worker.

Runs either embedded in the API process (`WORKER_EMBEDDED=1`, the default) or standalone via
`python -m slm_rag_eval.service.worker`. Both entry points share `EvaluationWorker`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable

from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.llm import build_client
from slm_rag_eval.llm.client import LLMClient
from slm_rag_eval.metrics.registry import evaluate
from slm_rag_eval.service.db import (
    Job,
    SessionFactory,
    claim_next_queued_job,
    create_engine,
    create_session_factory,
    create_tables,
    fail_job,
    finish_job,
)
from slm_rag_eval.service.logging import configure_logging, job_id_var
from slm_rag_eval.service.schemas import StoredJobRequest

logger = logging.getLogger(__name__)

JudgeFactory = Callable[[], LLMClient]

ATTEMPTS_PER_JOB = 2
"""One initial attempt plus one retry, then the job is marked `error`."""


class EvaluationWorker:
    """Claims queued jobs and evaluates them, `worker_concurrency` at a time."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        settings: Settings,
        judge_factory: JudgeFactory | None = None,
        poll_interval_s: float = 0.25,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._judge_factory = judge_factory or (lambda: build_client(settings))
        self._poll_interval_s = poll_interval_s
        self._semaphore = asyncio.Semaphore(settings.worker_concurrency)
        self._judge: LLMClient | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def run_forever(self) -> None:
        """Poll for work until cancelled. `aclose` drains whatever was still in flight."""
        while True:
            claimed = await self.run_once()
            if not claimed:
                await asyncio.sleep(self._poll_interval_s)

    async def run_once(self) -> bool:
        """Claim at most one job and start processing it. True when a job was claimed."""
        await self._semaphore.acquire()
        job = await claim_next_queued_job(self._session_factory)
        if job is None:
            self._semaphore.release()
            return False

        task = asyncio.create_task(self._process_and_release(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def aclose(self) -> None:
        """Finish in-flight jobs and close the judge client this worker created."""
        await self._drain()
        judge = self._judge
        self._judge = None
        aclose = getattr(judge, "aclose", None)
        if aclose is not None:
            await aclose()

    async def _drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _process_and_release(self, job: Job) -> None:
        token = job_id_var.set(job.id)
        try:
            await self._process(job)
        finally:
            job_id_var.reset(token)
            self._semaphore.release()

    async def _process(self, job: Job) -> None:
        try:
            stored = StoredJobRequest.model_validate(job.request_json)
        except ValueError as exc:  # Unreadable payload: no retry can fix it.
            await fail_job(self._session_factory, job.id, error=f"Invalid stored request: {exc}")
            return

        last_error: Exception | None = None
        for attempt in range(1, ATTEMPTS_PER_JOB + 1):
            try:
                result = await evaluate(
                    stored.request,
                    judge=self._get_judge(),
                    metrics=stored.options.metrics,
                    k=stored.options.k,
                    strict=stored.options.strict,
                    # The stored request was already masked at the API boundary, so masking it
                    # again would only re-detect placeholders. See README "Privacy layer".
                    settings=self._settings.model_copy(update={"privacy_mode": "off"}),
                )
            except Exception as exc:
                # Any judge or metric failure is a job failure; the job carries the message.
                last_error = exc
                logger.warning(
                    "job %s attempt %d/%d failed: %s", job.id, attempt, ATTEMPTS_PER_JOB, exc
                )
                continue
            await finish_job(
                self._session_factory, job.id, result_json=result.model_dump(mode="json")
            )
            return

        await fail_job(self._session_factory, job.id, error=str(last_error))

    def _get_judge(self) -> LLMClient:
        if self._judge is None:
            self._judge = self._judge_factory()
        return self._judge


async def run_standalone(settings: Settings | None = None) -> None:
    """Entry point for `python -m slm_rag_eval.service.worker`."""
    runtime_settings = settings or get_settings()
    engine = create_engine(runtime_settings.database_url)
    worker: EvaluationWorker | None = None
    try:
        await create_tables(engine)
        worker = EvaluationWorker(
            create_session_factory(engine),
            settings=runtime_settings,
        )
        logger.info("worker started (concurrency=%d)", runtime_settings.worker_concurrency)
        await worker.run_forever()
    finally:
        if worker is not None:
            await worker.aclose()
        await engine.dispose()


def main() -> None:
    """Console entry point; Ctrl-C stops the loop without a traceback."""
    configure_logging()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_standalone())


if __name__ == "__main__":  # pragma: no cover
    main()
