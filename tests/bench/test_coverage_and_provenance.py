"""Numbers are only meaningful next to the denominator and the configuration they came from.

Three ways a report used to overstate itself, each pinned here:

* a sample the judge failed on wrote no row, so it left the denominator silently;
* a row with no human label was read as a verified non-hallucination, so unlabeled CLI
  output scored as true negatives;
* two runs that differed only in privacy mode were pooled into one series, which averages
  the privacy ablation away instead of measuring it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slm_rag_eval.bench.analyze import (
    analyze,
    analyze_agreement,
    analyze_judge,
    assign_series,
    load_provenance,
    load_rows,
)
from slm_rag_eval.bench.run import (
    IncompatibleResumeError,
    run_benchmark,
    run_environment,
    run_id,
    run_identity,
)
from slm_rag_eval.core.config import Settings
from tests.conftest import FakeLLMClient
from tests.support import claim_list, verdict_batch

FIXTURE_DATA = Path(__file__).resolve().parents[1] / "data"


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {"privacy_mode": "off", "model": "synthetic-slm"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": "s1",
        "dataset": "d",
        "judge": "slm",
        "model": "m",
        "label_hallucinated": False,
        "scores": {"faithfulness": 1.0},
        "verdicts": [],
        "latency_ms": 10.0,
        "usage": {"total_tokens": 10},
    }
    row.update(overrides)
    return row


def _write(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if manifest is not None:
        path.with_name(f"{path.stem}.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return path


# --- coverage -----------------------------------------------------------------------


def test_failed_samples_are_counted_and_never_scored(tmp_path: Path) -> None:
    """A failed sample is a row with null scores and an error, not a missing row."""
    failed = [
        _row(sample_id=f"s{n}", scores={"faithfulness": None}, error="RuntimeError: boom")
        for n in (3, 4, 5)
    ]
    rows_path = _write(
        tmp_path / "d_slm.jsonl",
        [_row(sample_id="s1"), _row(sample_id="s2", label_hallucinated=True), *failed],
        manifest={"run_id": "abc123", "failure_count": 3, "failures": []},
    )

    report = analyze_judge(load_rows([rows_path]))

    assert report.scored == 2, "a failed sample must never count as scored"
    assert report.unscored == 3
    assert report.failed == 3, "the three failures must stay visible in the denominator"


def test_a_missing_manifest_reports_no_failures_rather_than_guessing(tmp_path: Path) -> None:
    rows_path = _write(tmp_path / "d_slm.jsonl", [_row()])

    provenance = load_provenance(rows_path)

    assert provenance.run_id is None
    assert provenance.failure_count == 0


def test_coverage_columns_reach_the_summary(tmp_path: Path) -> None:
    rows_path = _write(
        tmp_path / "d_slm.jsonl",
        [_row(sample_id="s1"), _row(sample_id="s2", label_hallucinated=True)],
        manifest={"run_id": "abc123", "failure_count": 2, "failures": []},
    )

    analyze([rows_path], tmp_path / "report")
    summary = (tmp_path / "report" / "summary.md").read_text()

    assert "Unlabeled" in summary
    assert "Failed" in summary


# --- labels -------------------------------------------------------------------------


def test_a_missing_label_is_unknown_rather_than_not_hallucinated(tmp_path: Path) -> None:
    """`rageval batch` writes rows with no label at all. They are not negatives."""
    unlabeled = _row()
    del unlabeled["label_hallucinated"]
    rows_path = _write(tmp_path / "d_cli.jsonl", [unlabeled])

    frame = load_rows([rows_path])

    assert frame["label_hallucinated"].isna().all()


def test_unlabeled_rows_are_excluded_from_quality_metrics(tmp_path: Path) -> None:
    unlabeled = _row(sample_id="s2", scores={"faithfulness": 0.0})
    del unlabeled["label_hallucinated"]
    rows_path = _write(tmp_path / "d_slm.jsonl", [_row(sample_id="s1"), unlabeled])

    report = analyze_judge(load_rows([rows_path]))

    assert report.scored == 2
    assert report.unlabeled == 1
    # Only s1 is labeled, and it is a negative: one negative, no positives.
    assert (report.positives, report.negatives) == (0, 1)


def test_entirely_unlabeled_output_reports_no_best_threshold(tmp_path: Path) -> None:
    """With no ground truth there is no detection quality to report — not a perfect score."""
    rows = []
    for index in (1, 2):
        row = _row(sample_id=f"s{index}", scores={"faithfulness": 0.5})
        del row["label_hallucinated"]
        rows.append(row)
    rows_path = _write(tmp_path / "d_cli.jsonl", rows)

    report = analyze_judge(load_rows([rows_path]))

    assert report.best is None
    assert report.roc_auc is None
    assert report.unlabeled == 2


# --- undefined statistics -----------------------------------------------------------


def test_kappa_is_not_reported_when_neither_judge_varies(tmp_path: Path) -> None:
    """Two constant, identical call vectors make kappa undefined, not perfect."""
    rows = [
        _row(sample_id=f"s{index}", judge=judge, model=judge, label_hallucinated=index > 2)
        for judge in ("slm", "cloud")
        for index in (1, 2, 3, 4)
    ]
    rows_path = _write(tmp_path / "d_both.jsonl", rows)
    frame = assign_series(load_rows([rows_path]))
    reports = {
        series: analyze_judge(frame[frame["series"] == series])
        for series in ("cloud", "slm")
    }

    agreements = analyze_agreement(frame, reports)

    assert len(agreements) == 1
    # Every faithfulness is 1.0, so at any threshold both judges call everything the same
    # way and neither vector varies.
    assert agreements[0].cohen_kappa is None


# --- provenance ---------------------------------------------------------------------


def test_the_run_fingerprint_covers_the_settings_that_change_what_a_score_means() -> None:
    base = _settings()
    identity = run_identity("d", "slm", "m", ["faithfulness"], base)
    baseline = run_id(identity, run_environment(base))

    for changed in (
        _settings(privacy_score_threshold=0.9),
        _settings(privacy_entities=["PERSON"]),
        _settings(max_tokens=512),
        _settings(base_url="http://elsewhere/v1"),
    ):
        assert run_id(identity, run_environment(changed)) != baseline


async def test_resume_refuses_when_only_the_token_budget_changed(tmp_path: Path) -> None:
    """`max_tokens` decides whether long samples truncate, so it is not a free parameter."""
    samples = _load_samples()
    await _run(tmp_path, samples, _settings())

    with pytest.raises(IncompatibleResumeError, match="different configuration"):
        await _run(tmp_path, samples, _settings(max_tokens=256))


async def test_resume_refuses_when_only_the_detector_configuration_changed(
    tmp_path: Path,
) -> None:
    samples = _load_samples()
    await _run(tmp_path, samples, _settings())

    with pytest.raises(IncompatibleResumeError, match="different configuration"):
        await _run(tmp_path, samples, _settings(privacy_score_threshold=0.95))


async def test_the_manifest_records_the_fingerprint_and_the_backend_model(
    tmp_path: Path,
) -> None:
    samples = _load_samples()

    manifest = await _run(tmp_path, samples, _settings())

    assert manifest["run_id"] == run_id(manifest["run_identity"], manifest["run_environment"])
    # The configured name is "synthetic-slm"; FakeLLMClient answers as "fake".
    assert manifest["models_returned"] == ["fake"]
    assert manifest["git_dirty"] in (True, False, None)


def test_two_runs_differing_only_in_privacy_mode_stay_separate_series(tmp_path: Path) -> None:
    """The privacy ablation is exactly this comparison; pooling it would erase the effect."""
    masked = _write(
        tmp_path / "d_slm.jsonl",
        [_row(sample_id="s1"), _row(sample_id="s2", label_hallucinated=True)],
        manifest={"run_id": "mask00", "failure_count": 0, "failures": []},
    )
    unmasked = _write(
        tmp_path / "d_slm_off.jsonl",
        [_row(sample_id="s1"), _row(sample_id="s2", label_hallucinated=True)],
        manifest={"run_id": "off000", "failure_count": 0, "failures": []},
    )

    frame = assign_series(load_rows([masked, unmasked]))

    assert set(frame["series"]) == {"slm#mask00", "slm#off000"}


def test_one_run_keeps_its_plain_judge_name(tmp_path: Path) -> None:
    """The common case must read exactly as before, with no fingerprint noise."""
    rows_path = _write(
        tmp_path / "d_slm.jsonl",
        [_row(sample_id="s1")],
        manifest={"run_id": "abc123", "failure_count": 0, "failures": []},
    )

    frame = assign_series(load_rows([rows_path]))

    assert set(frame["series"]) == {"slm"}


# --- helpers ------------------------------------------------------------------------


def _load_samples() -> list[Any]:
    from slm_rag_eval.bench.datasets import load

    return load("halueval", 1, data_dir=FIXTURE_DATA)


async def _run(tmp_path: Path, samples: list[Any], settings: Settings) -> dict[str, Any]:
    judge = FakeLLMClient()
    for sample in samples:
        claim = sample.answer.replace('"', "'")
        judge.push(
            claim_list(claim),
            verdict_batch(claim),
        )
    return await run_benchmark(
        samples,
        judge=judge,
        dataset="halueval",
        judge_name="slm",
        model=settings.model,
        out_dir=tmp_path,
        settings=settings,
        metrics=["faithfulness"],
    )
