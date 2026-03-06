"""Runtime configuration. Extended by tasks M01 (backend), M03 (metrics), M05 (service)."""

from __future__ import annotations

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


def get_settings() -> Settings:
    """Return a fresh Settings instance (kept simple; caching added if needed)."""
    return Settings()
