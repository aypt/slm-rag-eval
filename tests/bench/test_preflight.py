"""The preflight exists to be trusted when it says "go", so its verdicts are pinned here.

Its whole value is that a FAIL costs seconds instead of a rented GPU-hour. A preflight
that passes a configuration the real run would choke on is worse than none at all, so the
tests below are mostly about it *failing* for the right reasons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from slm_rag_eval.bench.datasets import load
from slm_rag_eval.bench.preflight import (
    FAIL,
    OK,
    WARN,
    check_configuration,
    check_context_budget,
    check_dataset,
    check_endpoint,
    check_live_round_trip,
    check_model_available,
    check_output_dir,
    check_privacy,
)
from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm.client import LLMResponse

FIXTURE_DATA = Path(__file__).resolve().parents[1] / "data"
CLAIM = "The probe carried three instruments."


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "privacy_mode": "off",
        "model": "stub-judge",
        "base_url": "http://judge.test/v1",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


# --- configuration ------------------------------------------------------------------


def test_an_empty_metric_selection_fails_configuration() -> None:
    assert check_configuration(_settings(), []).status == FAIL


def test_the_configuration_line_carries_everything_the_report_must_state() -> None:
    check = check_configuration(_settings(k=3, max_tokens=4096), ["faithfulness"])

    assert check.status == OK
    for expected in ("stub-judge", "k=3", "max_tokens=4096", "privacy=off"):
        assert expected in check.detail


# --- endpoint and model -------------------------------------------------------------


def test_an_unreachable_endpoint_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", boom)

    check, served = check_endpoint(_settings())

    assert check.status == FAIL
    assert served == []


def test_a_model_that_was_never_pulled_fails_with_a_near_miss_hint() -> None:
    """The compose race: the endpoint is up but `ollama pull` has not finished."""
    check = check_model_available(_settings(model="qwen2.5:7b-instruct"), ["qwen2.5:3b"])

    assert check.status == FAIL
    assert "Did you mean" in check.detail


def test_a_served_model_passes() -> None:
    assert check_model_available(_settings(), ["stub-judge"]).status == OK


# --- dataset ------------------------------------------------------------------------


def test_a_missing_dataset_fails_with_the_download_instruction(tmp_path: Path) -> None:
    check, samples = check_dataset("ragtruth", None, tmp_path)

    assert check.status == FAIL
    assert "download_ragtruth" in check.detail
    assert samples == []


def test_a_balanced_dataset_reports_its_label_split() -> None:
    check, samples = check_dataset("halueval", 4, FIXTURE_DATA)

    assert check.status == OK
    assert len(samples) == 4
    assert "hallucinated" in check.detail


# --- budgets and privacy ------------------------------------------------------------


def test_the_context_budget_check_reports_the_longest_sample() -> None:
    samples = load("halueval", 4, data_dir=FIXTURE_DATA)

    check = check_context_budget(samples, _settings())

    assert check.status == OK
    assert "max=" in check.detail


def test_privacy_off_is_a_warning_because_masking_is_not_active() -> None:
    check = check_privacy(_settings(privacy_mode="off"))

    assert check.status == WARN
    assert "NOT active" in check.detail


# --- output directory ---------------------------------------------------------------


def test_an_incompatible_existing_output_directory_fails(tmp_path: Path) -> None:
    """Exactly the mistake that silently mixes two judges' rows into one file."""
    (tmp_path / "halueval_slm.jsonl").write_text('{"sample_id": "s1"}\n', encoding="utf-8")
    (tmp_path / "halueval_slm.manifest.json").write_text(
        json.dumps(
            {
                "run_identity": {
                    "dataset": "halueval",
                    "judge": "slm",
                    "model": "a-different-model",
                    "metrics": ["faithfulness"],
                    "k": 1,
                    "strict": True,
                    "privacy_mode": "off",
                },
                "run_environment": {},
            }
        ),
        encoding="utf-8",
    )

    check = check_output_dir(
        tmp_path, "halueval", "slm", _settings(), ["faithfulness"], "stub-judge"
    )

    assert check.status == FAIL
    assert "different configuration" in check.detail


def test_a_clean_output_directory_passes(tmp_path: Path) -> None:
    check = check_output_dir(
        tmp_path / "new", "halueval", "slm", _settings(), ["faithfulness"], "stub-judge"
    )

    assert check.status == OK


# --- live round trip ----------------------------------------------------------------


class _Judge:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        if self.error is not None:
            raise self.error
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        text = (
            json.dumps([{"claim": CLAIM, "verdict": "supported", "reason": "The context agrees."}])
            if "Claims to verify" in joined
            else json.dumps({"claims": [CLAIM]})
        )
        return LLMResponse(text=text, usage={"prompt_tokens": 10, "completion_tokens": 4})


async def test_a_working_judge_reports_latency_and_a_run_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slm_rag_eval.bench import preflight

    monkeypatch.setattr(
        preflight, "build_judge", lambda judge, settings: (_Judge(), settings)
    )
    samples = load("halueval", 2, data_dir=FIXTURE_DATA)

    checks = await check_live_round_trip(samples, _settings(), ["faithfulness"], "slm", 2)

    by_name = {check.name: check for check in checks}
    assert by_name["live round trip"].status == OK
    assert "2/2 scored" in by_name["live round trip"].detail
    assert "projected 200-sample run" in by_name


async def test_truncation_is_reported_on_its_own_line_even_when_nothing_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cheapest fix must never be buried inside a generic "nothing scored" message."""
    from slm_rag_eval.bench import preflight
    from slm_rag_eval.llm.errors import TruncatedResponseError

    monkeypatch.setattr(
        preflight,
        "build_judge",
        lambda judge, settings: (_Judge(TruncatedResponseError("hit the budget")), settings),
    )
    samples = load("halueval", 2, data_dir=FIXTURE_DATA)

    checks = await check_live_round_trip(samples, _settings(), ["faithfulness"], "slm", 2)

    by_name = {check.name: check for check in checks}
    assert by_name["live round trip"].status == FAIL
    assert by_name["truncation"].status == FAIL
    assert "SLMEVAL_MAX_TOKENS" in by_name["truncation"].detail


async def test_a_judge_that_never_returns_valid_json_fails_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slm_rag_eval.bench import preflight

    monkeypatch.setattr(
        preflight,
        "build_judge",
        lambda judge, settings: (_Judge(ValueError("Malformed body")), settings),
    )
    samples = load("halueval", 1, data_dir=FIXTURE_DATA)

    checks = await check_live_round_trip(samples, _settings(), ["faithfulness"], "slm", 1)

    assert checks[0].status == FAIL
    assert "0/1 scored" in checks[0].detail
