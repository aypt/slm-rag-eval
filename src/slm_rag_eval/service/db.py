"""Async persistence for evaluation jobs (SQLAlchemy 2).

The default backend is SQLite via aiosqlite; `DATABASE_URL` switches it to Postgres without
code changes. Job claiming uses a conditional UPDATE rather than `SELECT ... FOR UPDATE`, so
the same statement is correct for single-writer SQLite and for Postgres.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from sqlalchemy import JSON, CursorResult, DateTime, String, Text, select, update
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JobStatus = Literal["queued", "running", "done", "error"]

SessionFactory = async_sessionmaker[AsyncSession]


def utcnow() -> datetime:
    """Timezone-aware now; stored as UTC on every backend."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for the service's tables."""


class Job(Base):
    """One submitted evaluation and its lifecycle state."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default="queued")
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


def create_engine(database_url: str) -> AsyncEngine:
    """Create the async engine, creating the SQLite parent directory when one is needed."""
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database:
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    return create_async_engine(database_url, future=True)


def create_session_factory(engine: AsyncEngine) -> SessionFactory:
    """Session factory whose objects stay usable after commit."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_tables(engine: AsyncEngine) -> None:
    """Create missing tables. Safe to call on every startup."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def create_job(session_factory: SessionFactory, request_json: dict[str, Any]) -> Job:
    """Insert a queued job. `request_json` is already masked when privacy mode is `mask`."""
    job = Job(id=str(uuid.uuid4()), status="queued", request_json=request_json)
    async with session_factory() as session:
        session.add(job)
        await session.commit()
    return job


async def get_job(session_factory: SessionFactory, job_id: str) -> Job | None:
    """Load one job, or None when the id is unknown."""
    async with session_factory() as session:
        return await session.get(Job, job_id)


async def claim_next_queued_job(session_factory: SessionFactory) -> Job | None:
    """Atomically move the oldest queued job to `running` and return it.

    The conditional UPDATE is the claim: a worker that loses the race sees `rowcount == 0`
    and moves on to the next candidate instead of processing a job twice.
    """
    async with session_factory() as session:
        while True:
            candidate = await session.execute(
                select(Job.id)
                .where(Job.status == "queued")
                .order_by(Job.created_at, Job.id)
                .limit(1)
            )
            job_id = candidate.scalar_one_or_none()
            if job_id is None:
                return None

            # A DML execute returns a CursorResult; only that subclass carries `rowcount`.
            claimed = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(Job)
                    .where(Job.id == job_id, Job.status == "queued")
                    .values(status="running", updated_at=utcnow())
                ),
            )
            await session.commit()
            if claimed.rowcount == 1:
                return await session.get(Job, job_id)


async def finish_job(
    session_factory: SessionFactory,
    job_id: str,
    *,
    result_json: dict[str, Any],
) -> None:
    """Store a successful result and mark the job done."""
    await _update_job(session_factory, job_id, status="done", result_json=result_json, error=None)


async def fail_job(session_factory: SessionFactory, job_id: str, *, error: str) -> None:
    """Mark the job failed, keeping the message for the API to report."""
    await _update_job(session_factory, job_id, status="error", result_json=None, error=error)


async def _update_job(
    session_factory: SessionFactory,
    job_id: str,
    *,
    status: JobStatus,
    result_json: dict[str, Any] | None,
    error: str | None,
) -> None:
    async with session_factory() as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status=status, result_json=result_json, error=error, updated_at=utcnow())
        )
        await session.commit()
