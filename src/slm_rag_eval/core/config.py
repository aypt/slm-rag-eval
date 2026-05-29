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

    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible judge endpoint.",
    )
    api_key: str | None = Field(
        default=None, description="Bearer token; leave unset for a local Ollama."
    )
    model: str = Field(default="qwen2.5:7b-instruct", description="Judge model name.")
    timeout_s: float = Field(default=120.0, gt=0, description="Per-request timeout in seconds.")
    max_retries: int = Field(
        default=3, ge=1, description="Attempts for transient transport failures."
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Completion token budget per judge call; raise it for long contexts.",
    )
    enabled_metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness"],
        description="Metrics the registry runs by default (JSON list).",
    )
    k: int = Field(default=1, ge=1, description="Self-consistency verification runs.")
    strict: bool = Field(
        default=True, description="Count uncertain verdicts as unsupported."
    )
    privacy_mode: Literal["mask", "off"] = Field(
        default="mask", description="`mask` sanitizes every request before it leaves the process."
    )
    privacy_entities: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ENTITIES),
        description="Presidio entity types to detect (JSON list).",
    )
    privacy_score_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Minimum detector confidence."
    )

    # Service settings keep their conventional un-prefixed names (DATABASE_URL is what a
    # Postgres deployment already sets), with the SLMEVAL_-prefixed spelling also accepted.
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/jobs.db",
        validation_alias=AliasChoices("DATABASE_URL", "SLMEVAL_DATABASE_URL"),
        description="Job store; SQLite by default, Postgres in the compose deployment.",
    )
    worker_embedded: bool = Field(
        default=True,
        validation_alias=AliasChoices("WORKER_EMBEDDED", "SLMEVAL_WORKER_EMBEDDED"),
        description="Run the worker inside the API process.",
    )
    worker_concurrency: int = Field(
        default=2,
        ge=1,
        validation_alias=AliasChoices("WORKER_CONCURRENCY", "SLMEVAL_WORKER_CONCURRENCY"),
        description="Jobs evaluated in parallel per worker.",
    )


def get_settings() -> Settings:
    """Return a fresh Settings instance (kept simple; caching added if needed)."""
    return Settings()
