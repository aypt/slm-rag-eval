from __future__ import annotations

import pytest
from pydantic import ValidationError

from slm_rag_eval.core.config import Settings


def test_metric_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.enabled_metrics == ["faithfulness"]
    assert settings.k == 1
    assert settings.strict is True
    assert settings.privacy_mode == "mask"


def test_metric_settings_can_be_overridden_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLMEVAL_ENABLED_METRICS", '["faithfulness", "relevance"]')
    monkeypatch.setenv("SLMEVAL_K", "3")
    monkeypatch.setenv("SLMEVAL_STRICT", "false")
    monkeypatch.setenv("SLMEVAL_PRIVACY_MODE", "off")

    settings = Settings(_env_file=None)

    assert settings.enabled_metrics == ["faithfulness", "relevance"]
    assert settings.k == 3
    assert settings.strict is False
    assert settings.privacy_mode == "off"


def test_privacy_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Settings(privacy_mode="redact", _env_file=None)  # type: ignore[arg-type]
