"""FastAPI application: submit an evaluation, poll it, read the result.

Masking happens here, at the trust boundary, not in the worker: the request is sanitized
before the job row is written, so raw PII never reaches the database, the judge, or the
stored result. The placeholder mapping is dropped immediately — it is never logged or
persisted (AGENTS.md rule 2), so results served by this API stay masked.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncEngine

from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.metrics.registry import RequestSanitizer, sanitize_for_judge
from slm_rag_eval.service.db import (
    SessionFactory,
    create_engine,
    create_job,
    create_session_factory,
    create_tables,
    get_job,
)
from slm_rag_eval.service.schemas import (
    EvaluationSubmission,
    JobStatusResponse,
    StoredJobRequest,
    SubmitResponse,
)
from slm_rag_eval.service.worker import EvaluationWorker, JudgeFactory


def get_app_settings(request: Request) -> Settings:
    """Service configuration resolved once at startup."""
    settings: Settings = request.app.state.settings
    return settings


def get_session_factory(request: Request) -> SessionFactory:
    """Session factory bound to this app's engine."""
    session_factory: SessionFactory = request.app.state.session_factory
    return session_factory


def get_sanitizer(request: Request) -> RequestSanitizer | None:
    """Detector override, if one was injected; otherwise the shared default is used."""
    sanitizer: RequestSanitizer | None = request.app.state.sanitizer
    return sanitizer


AppSettings = Annotated[Settings, Depends(get_app_settings)]
Sessions = Annotated[SessionFactory, Depends(get_session_factory)]
InjectedSanitizer = Annotated["RequestSanitizer | None", Depends(get_sanitizer)]


def create_app(
    *,
    settings: Settings | None = None,
    engine: AsyncEngine | None = None,
    judge_factory: JudgeFactory | None = None,
    sanitizer: RequestSanitizer | None = None,
    worker_poll_interval_s: float = 0.25,
) -> FastAPI:
    """Build the application. Every collaborator is injectable so tests stay offline."""
    runtime_settings = settings or get_settings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_engine = engine is None
        active_engine = engine or create_engine(runtime_settings.database_url)
        session_factory = create_session_factory(active_engine)
        await create_tables(active_engine)

        app.state.settings = runtime_settings
        app.state.session_factory = session_factory
        app.state.sanitizer = sanitizer

        worker: EvaluationWorker | None = None
        worker_task: asyncio.Task[None] | None = None
        if runtime_settings.worker_embedded:
            worker = EvaluationWorker(
                session_factory,
                settings=runtime_settings,
                judge_factory=judge_factory,
                poll_interval_s=worker_poll_interval_s,
            )
            worker_task = asyncio.create_task(worker.run_forever())
        app.state.worker = worker
        try:
            yield
        finally:
            if worker_task is not None:
                worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_task
            if worker is not None:
                await worker.aclose()
            if owns_engine:
                await active_engine.dispose()

    app = FastAPI(title="slm-rag-eval", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness probe used by container healthchecks."""
        return {"status": "ok"}

    @app.post(
        "/v1/evaluations",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=SubmitResponse,
    )
    async def submit_evaluation(
        submission: EvaluationSubmission,
        app_settings: AppSettings,
        session_factory: Sessions,
        request_sanitizer: InjectedSanitizer,
    ) -> SubmitResponse:
        """Queue one evaluation and return its job id."""
        options = submission.options.resolve(app_settings)
        judge_request, _mapping = sanitize_for_judge(
            submission.to_eval_request(),
            settings=app_settings.model_copy(update={"privacy_mode": options.privacy_mode}),
            sanitizer=request_sanitizer,
        )
        # `_mapping` is intentionally discarded: persisting it would put the raw PII back into
        # storage, which is exactly what privacy mode exists to prevent.
        stored = StoredJobRequest(request=judge_request, options=options)
        job = await create_job(session_factory, stored.model_dump(mode="json"))
        return SubmitResponse(job_id=job.id)

    @app.get("/v1/evaluations/{job_id}", response_model=JobStatusResponse)
    async def read_evaluation(
        job_id: str,
        session_factory: Sessions,
    ) -> JobStatusResponse:
        """Report job status, plus the result or the failure message once it is settled."""
        job = await get_job(session_factory, job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown job: {job_id}"
            )
        return JobStatusResponse.model_validate(
            {
                "job_id": job.id,
                "status": job.status,
                "result": job.result_json,
                "error": job.error,
            }
        )

    return app


app = create_app()
