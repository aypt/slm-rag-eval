"""FastAPI application. Evaluation endpoints are implemented in task M05."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="slm-rag-eval", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe used by container healthchecks."""
    return {"status": "ok"}
