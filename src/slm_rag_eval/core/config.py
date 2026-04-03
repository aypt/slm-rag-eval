"""Runtime configuration for model, metric, privacy, and service behavior."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from slm_rag_eval.privacy.sanitizer import DEFAULT_ENTITIES


class Settings(BaseSettings):
    """All settings are overridable via SLMEVAL_* environment variables or a .env file."""

    # populate_by_name so aliased fields (database_url, worker_*) can also be set as keyword
    # arguments, which is how tests and the app factory build an explicit Settings object.
    model_config = SettingsConfigDict(
        env_prefix="SLMEVAL_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    model: str = "qwen2.5:7b-instruct"
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=1)
    enabled_metrics: list[str] = Field(default_factory=lambda: ["faithfulness"])
    k: int = Field(default=1, ge=1)
    strict: bool = True
    privacy_mode: Literal["mask", "off"] = "mask"
    privacy_entities: list[str] = Field(default_factory=lambda: list(DEFAULT_ENTITIES))
    privacy_score_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # Service settings keep their conventional un-prefixed names (DATABASE_URL is what a
    # Postgres deployment already sets), with the SLMEVAL_-prefixed spelling also accepted.
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/jobs.db",
        validation_alias=AliasChoices("DATABASE_URL", "SLMEVAL_DATABASE_URL"),
    )
    worker_embedded: bool = Field(
        default=True,
        validation_alias=AliasChoices("WORKER_EMBEDDED", "SLMEVAL_WORKER_EMBEDDED"),
    )
    worker_concurrency: int = Field(
        default=2,
        ge=1,
        validation_alias=AliasChoices("WORKER_CONCURRENCY", "SLMEVAL_WORKER_CONCURRENCY"),
    )


def get_settings() -> Settings:
    """Return a fresh Settings instance (kept simple; caching added if needed)."""
    return Settings()
