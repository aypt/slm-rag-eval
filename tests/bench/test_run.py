from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from slm_rag_eval.bench.datasets import LabeledSample, load
from slm_rag_eval.bench.run import (
    IncompatibleResumeError,
    app,
    apply_privacy_mode,
    build_judge,
    output_paths,
    run_benchmark,
)
from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm.client import LLMResponse
from tests.conftest import FakeLLMClient

FIXTURE_DATA = Path(__file__).resolve().parents[1] / "data"


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {"privacy_mode": "off", "model": "synthetic-slm"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _samples(count: int = 2) -> list[LabeledSample]:
    return load("halueval", count, data_dir=FIXTURE_DATA)


def _script_faithfulness(judge: FakeLLMClient, samples: list[LabeledSample]) -> None:
    """One extraction and one verification response per sample."""
    for sample in samples:
        claim = sample.answer.replace('"', "'")
        judge.push(
            json.dumps({"claims": [claim]}),
            json.dumps(
                [{"claim": claim, "verdict": "supported", "reason": "The context agrees."}]
            ),
        )


async def test_run_writes_one_row_per_sample_plus_a_manifest(tmp_path: Path) -> None:
    samples = _samples(2)
    judge = FakeLLMClient()
    _script_faithfulness(judge, samples)

    manifest = await run_benchmark(
        samples,
        judge=judge,
        dataset="halueval",
        judge_name="slm",
        model="synthetic-slm",
        out_dir=tmp_path,
        settings=_settings(),
        metrics=["faithfulness"],
    )

    rows_path, manifest_path = output_paths(tmp_path, "halueval", "slm")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]

    assert [row["sample_id"] for row in rows] == [sample.id for sample in samples]
    first = rows[0]
    assert set(first) == {
        "sample_id",
        "dataset",
        "judge",
        "model",
        "label_hallucinated",
        "scores",
        "verdicts",
        "latency_ms",
        "usage",
    }
    assert first["dataset"] == "halueval"
    assert first["judge"] == "slm"
    assert first["model"] == "synthetic-slm"
    assert first["label_hallucinated"] is False
    assert first["scores"]["faithfulness"] == 1.0
    assert first["verdicts"][0]["verdict"] == "supported"
    assert first["latency_ms"] >= 0

    stored_manifest = json.loads(manifest_path.read_text())
    assert stored_manifest == manifest
    assert manifest["samples_written"] == 2
    assert manifest["failure_count"] == 0
    assert manifest["params"]["privacy_mode"] == "off"
    assert manifest["timestamp_utc"].endswith("+00:00")


async def test_rerunning_the_same_output_skips_completed_samples(tmp_path: Path) -> None:
    samples = _samples(2)
    first_judge = FakeLLMClient()
    _script_faithfulness(first_judge, samples)
    settings = _settings()

    await run_benchmark(
        samples,
        judge=first_judge,
        dataset="halueval",
        judge_name="slm",
        model="synthetic-slm",
        out_dir=tmp_path,
        settings=settings,
        metrics=["faithfulness"],
    )

    extended = _samples(3)
    resumed_judge = FakeLLMClient()
    _script_faithfulness(resumed_judge, extended[2:])

    manifest = await run_benchmark(
        extended,
        judge=resumed_judge,
        dataset="halueval",
        judge_name="slm",
        model="synthetic-slm",
        out_dir=tmp_path,
        settings=settings,
        metrics=["faithfulness"],
    )

    rows_path, _ = output_paths(tmp_path, "halueval", "slm")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]

    assert manifest["samples_skipped"] == 2
    assert manifest["samples_written"] == 1
    assert [row["sample_id"] for row in rows] == [sample.id for sample in extended]
    # Only the new sample cost judge calls.
    assert len(resumed_judge.calls) == 2


async def test_one_failing_sample_does_not_end_the_run(tmp_path: Path) -> None:
    samples = _samples(2)

    class FailFirstJudge(FakeLLMClient):
        """Fails the very first call, then behaves like the scripted fake."""

        failed_once = False

        async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
            if not self.failed_once:
                self.failed_once = True
                raise RuntimeError("backend exploded")
            return await super().complete(messages, **kwargs)

    judge = FailFirstJudge()
    _script_faithfulness(judge, samples[1:])

    manifest = await run_benchmark(
        samples,
        judge=judge,
        dataset="halueval",
        judge_name="slm",
        model="synthetic-slm",
        out_dir=tmp_path,
        settings=_settings(),
        metrics=["faithfulness"],
    )

    rows_path, _ = output_paths(tmp_path, "halueval", "slm")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]

    assert manifest["failure_count"] == 1
    assert manifest["failures"][0]["sample_id"] == samples[0].id
    assert "backend exploded" in manifest["failures"][0]["error"]
    assert manifest["samples_written"] == 1
    assert [row["sample_id"] for row in rows] == [samples[1].id]


def test_cloud_judge_requires_explicit_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUD_BASE_URL", raising=False)
    monkeypatch.delenv("CLOUD_MODEL", raising=False)

    with pytest.raises(ValueError, match="CLOUD_BASE_URL and CLOUD_MODEL"):
        build_judge("cloud", _settings())

    monkeypatch.setenv("CLOUD_BASE_URL", "https://judge.example.com/v1")
    monkeypatch.setenv("CLOUD_MODEL", "some-cloud-model")
    monkeypatch.setenv("CLOUD_API_KEY", "synthetic-key")

    _, cloud_settings = build_judge("cloud", _settings())

    assert cloud_settings.base_url == "https://judge.example.com/v1"
    assert cloud_settings.model == "some-cloud-model"
    assert cloud_settings.api_key == "synthetic-key"
    # The local judge is untouched by the cloud environment variables.
    _, slm_settings = build_judge("slm", _settings())
    assert slm_settings.model == "synthetic-slm"


def test_unknown_judge_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown judge: oracle"):
        build_judge("oracle", _settings())


def test_privacy_mode_override_is_validated_not_trusted() -> None:
    """A typo must never land in Settings: every non-`mask` value disables masking."""
    settings = _settings(privacy_mode="mask")

    assert apply_privacy_mode(settings, None).privacy_mode == "mask"
    assert apply_privacy_mode(settings, "off").privacy_mode == "off"
    assert apply_privacy_mode(_settings(privacy_mode="off"), "mask").privacy_mode == "mask"

    with pytest.raises(ValidationError):
        apply_privacy_mode(settings, "typo")
    with pytest.raises(ValidationError):
        apply_privacy_mode(settings, "MASK")


def test_privacy_mode_override_keeps_every_other_setting() -> None:
    settings = _settings(privacy_mode="mask", k=3, strict=False, model="synthetic-slm")

    overridden = apply_privacy_mode(settings, "off")

    assert (overridden.k, overridden.strict, overridden.model) == (3, False, "synthetic-slm")


def test_cli_rejects_an_unknown_privacy_mode() -> None:
    result = CliRunner().invoke(
        app, ["--privacy-mode", "typo", "--dataset", "halueval", "--data-dir", str(FIXTURE_DATA)]
    )

    assert result.exit_code != 0
    assert "'typo' is not one of 'mask', 'off'" in result.output.replace("\n", "")


async def _run(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Score one sample into tmp_path with an easily varied configuration."""
    samples = _samples(1)
    judge = FakeLLMClient()
    _script_faithfulness(judge, samples)
    kwargs: dict[str, Any] = {
        "dataset": "halueval",
        "judge_name": "slm",
        "model": "synthetic-slm",
        "settings": _settings(),
        "metrics": ["faithfulness"],
    }
    kwargs.update(overrides)
    return await run_benchmark(samples, judge=judge, out_dir=tmp_path, **kwargs)


async def test_resume_records_the_run_identity_and_contributing_commits(
    tmp_path: Path,
) -> None:
    manifest = await _run(tmp_path)

    assert manifest["run_identity"] == {
        "dataset": "halueval",
        "judge": "slm",
        "model": "synthetic-slm",
        "metrics": ["faithfulness"],
        "k": 1,
        "strict": True,
        "privacy_mode": "off",
    }
    assert manifest["git_shas"] == [manifest["git_sha"]]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "a-different-model"),
        ("metrics", ["faithfulness", "relevance"]),
        ("settings", None),  # replaced below with different scoring parameters
    ],
)
async def test_resume_refuses_a_changed_configuration(
    tmp_path: Path, field: str, value: Any
) -> None:
    """Appending rows scored under different settings would make the file unattributable."""
    await _run(tmp_path)

    override = {field: value} if field != "settings" else {"settings": _settings(k=3)}
    with pytest.raises(IncompatibleResumeError, match="different configuration"):
        await _run(tmp_path, **override)


async def test_resume_requires_a_manifest_it_can_check(tmp_path: Path) -> None:
    await _run(tmp_path)
    _, manifest_path = output_paths(tmp_path, "halueval", "slm")
    manifest_path.unlink()

    with pytest.raises(IncompatibleResumeError, match="cannot be verified"):
        await _run(tmp_path)


async def test_resume_accepts_an_unchanged_configuration(tmp_path: Path) -> None:
    first = await _run(tmp_path)
    second = await _run(tmp_path)

    assert second["samples_skipped"] == 1
    assert second["samples_written"] == 0
    assert second["run_identity"] == first["run_identity"]
