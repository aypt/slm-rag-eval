from __future__ import annotations

import pytest

from slm_rag_eval.core.config import Settings
from slm_rag_eval.privacy.sanitizer import DEFAULT_ENTITIES


def test_privacy_detector_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.privacy_entities == list(DEFAULT_ENTITIES)
    assert settings.privacy_score_threshold == 0.4


def test_privacy_detector_settings_can_be_overridden_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLMEVAL_PRIVACY_ENTITIES", '["EMAIL_ADDRESS", "IP_ADDRESS"]')
    monkeypatch.setenv("SLMEVAL_PRIVACY_SCORE_THRESHOLD", "0.7")

    settings = Settings(_env_file=None)

    assert settings.privacy_entities == ["EMAIL_ADDRESS", "IP_ADDRESS"]
    assert settings.privacy_score_threshold == 0.7
