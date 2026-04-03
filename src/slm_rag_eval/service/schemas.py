"""Wire and storage schemas for the evaluation service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from slm_rag_eval.core.config import Settings
from slm_rag_eval.core.schemas import EvalRequest, EvalResult

PrivacyMode = Literal["mask", "off"]


class EvaluationOptions(BaseModel):
    """Per-request overrides. Anything left unset falls back to `Settings`."""

    metrics: list[str] | None = None
    k: int | None = None
    strict: bool | None = None
    privacy_mode: PrivacyMode | None = None

    def resolve(self, settings: Settings) -> ResolvedOptions:
        """Fill every unset option from the service configuration."""
        return ResolvedOptions(
            metrics=list(self.metrics) if self.metrics is not None else list(
                settings.enabled_metrics
            ),
            k=self.k if self.k is not None else settings.k,
            strict=self.strict if self.strict is not None else settings.strict,
            privacy_mode=(
                self.privacy_mode if self.privacy_mode is not None else settings.privacy_mode
            ),
        )


class ResolvedOptions(BaseModel):
    """Options with no unset values left; this is what gets persisted and replayed."""

    metrics: list[str]
    k: int = Field(ge=1)
    strict: bool
    privacy_mode: PrivacyMode


class EvaluationSubmission(EvalRequest):
    """POST body: an `EvalRequest` plus optional per-request options."""

    options: EvaluationOptions = Field(default_factory=EvaluationOptions)

    def to_eval_request(self) -> EvalRequest:
        """The evaluation payload without the service-only options field."""
        return EvalRequest(question=self.question, answer=self.answer, contexts=self.contexts)


class StoredJobRequest(BaseModel):
    """Shape of `Job.request_json`.

    The request stored here is the one the judge will see: when privacy mode is `mask` it is
    already sanitized, so no raw PII is ever written to the database (M04 privacy invariant
    extended to persistence). The placeholder mapping is deliberately not part of this model.
    """

    request: EvalRequest
    options: ResolvedOptions


class SubmitResponse(BaseModel):
    """202 body for an accepted evaluation."""

    job_id: str
    status: Literal["queued"] = "queued"


class JobStatusResponse(BaseModel):
    """GET body for one job."""

    job_id: str
    status: Literal["queued", "running", "done", "error"]
    result: EvalResult | None = None
    error: str | None = None
