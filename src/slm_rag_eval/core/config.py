"""Runtime configuration. Extended by tasks M01 (backend), M03 (metrics), M05 (service)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings are overridable via SLMEVAL_* environment variables or a .env file."""

    model_config = SettingsConfigDict(env_prefix="SLMEVAL_", env_file=".env", extra="ignore")

    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    model: str = "qwen2.5:7b-instruct"
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=1)
    enabled_metrics: list[str] = Field(default_factory=lambda: ["faithfulness"])
    k: int = Field(default=1, ge=1)
    strict: bool = True
    # TODO(M04): apply masking before judge calls and persistence. Until then both modes
    # are accepted configuration values and intentionally follow the same code path.
    privacy_mode: Literal["mask", "off"] = "mask"


def get_settings() -> Settings:
    """Return a fresh Settings instance (kept simple; caching added if needed)."""
    return Settings()
