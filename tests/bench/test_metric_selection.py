"""A bad metric selection must stop the run, not fail every sample inside it.

`evaluate()` validates metric names, but it runs once per sample inside the benchmark's
fail-soft loop. A typo therefore produced 200 identical per-sample failures, an exit code
of 0, and a manifest that looked like a completed run. An empty selection was worse: it
succeeded and scored nothing at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from slm_rag_eval.bench import run as bench_run
from slm_rag_eval.bench.datasets import load
from slm_rag_eval.bench.run import output_paths, run_benchmark
from slm_rag_eval.core.config import Settings
from slm_rag_eval.metrics.registry import validate_metric_selection
from tests.conftest import FakeLLMClient
from tests.support import claim_list, verdict_batch

runner = CliRunner()
FIXTURE_DATA = Path(__file__).resolve().parents[1] / "data"


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {"privacy_mode": "off", "model": "synthetic-slm"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def test_an_empty_selection_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="No metrics selected"):
        validate_metric_selection([])


def test_an_unknown_metric_is_rejected_and_lists_what_is_available() -> None:
    with pytest.raises(ValueError, match="faithfulness"):
        validate_metric_selection(["faithfulnes"])


async def test_a_typo_stops_the_run_instead_of_failing_every_sample(tmp_path: Path) -> None:
    samples = load("halueval", 2, data_dir=FIXTURE_DATA)

    with pytest.raises(ValueError, match="Unknown metric"):
        await run_benchmark(
            samples,
            judge=FakeLLMClient(),
            dataset="halueval",
            judge_name="slm",
            model="synthetic-slm",
            out_dir=tmp_path,
            settings=_settings(),
            metrics=["faithfulnes"],
        )

    rows_path, manifest_path = output_paths(tmp_path, "halueval", "slm")
    assert not manifest_path.exists(), "a rejected run must not leave a manifest behind"
    assert not rows_path.exists()


def test_the_cli_rejects_a_typo_before_loading_the_dataset(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Nothing expensive should start: no dataset load, no client, no judge call."""
    loaded: list[str] = []
    monkeypatch.setattr(
        bench_run, "load", lambda *args, **kwargs: loaded.append("loaded") or []
    )
    monkeypatch.setattr(bench_run, "get_settings", _settings)

    result = runner.invoke(
        bench_run.app,
        ["--dataset", "halueval", "--metric", "faithfulnes", "--out", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert loaded == [], "the dataset was loaded before the metric name was checked"


def test_a_good_selection_still_runs(tmp_path: Path) -> None:
    """The guard must not get in the way of the normal path."""
    samples = load("halueval", 1, data_dir=FIXTURE_DATA)
    judge = FakeLLMClient()
    claim = samples[0].answer.replace('"', "'")
    judge.push(
        claim_list(claim),
        verdict_batch(claim),
    )

    import asyncio

    manifest = asyncio.run(
        run_benchmark(
            samples,
            judge=judge,
            dataset="halueval",
            judge_name="slm",
            model="synthetic-slm",
            out_dir=tmp_path,
            settings=_settings(),
            metrics=["faithfulness"],
        )
    )

    assert manifest["samples_written"] == 1
    assert manifest["failure_count"] == 0
