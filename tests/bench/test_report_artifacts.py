"""The report's evidence has to be produced correctly, not merely produced.

Covers the pieces added so a rented machine can generate every figure and table in one
pass: sampling protocol, held-out thresholds, kappa against gold, the privacy ablation,
error-analysis denominators, and environment capture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slm_rag_eval.bench import environment as environment_module
from slm_rag_eval.bench.analyze import (
    analyze,
    analyze_judge,
    analyze_privacy_ablation,
    assign_series,
    holdout_fold,
    load_rows,
    select_threshold_holdout,
    threshold_metrics,
)
from slm_rag_eval.bench.datasets import (
    LabeledSample,
    describe,
    select_split,
    shuffled,
    stratified_sample,
)
from slm_rag_eval.bench.errors import bucket_totals, collect_errors, sample_buckets
from slm_rag_eval.bench.experiment import plan_legs

FIXTURE_DATA = Path(__file__).resolve().parents[1] / "data"


def _sample(index: int, *, hallucinated: bool, task: str = "qa", split: str = "train") -> Any:
    return LabeledSample(
        id=f"s{index}",
        question="Q?",
        answer="A.",
        contexts=["ctx"],
        label_hallucinated=hallucinated,
        meta={"task_type": task, "split": split},
    )


# --- sampling protocol ---------------------------------------------------------------


def test_selecting_a_split_that_does_not_exist_fails_loudly() -> None:
    """Silently returning training data from a run labelled "test" is undetectable later."""
    samples = [_sample(i, hallucinated=False, split="train") for i in range(4)]

    with pytest.raises(ValueError, match="Available splits: train"):
        select_split(samples, "test")


def test_selecting_a_split_keeps_only_that_split() -> None:
    samples = [
        _sample(0, hallucinated=False, split="train"),
        _sample(1, hallucinated=True, split="test"),
    ]

    assert [s.id for s in select_split(samples, "test")] == ["s1"]


def test_the_same_seed_gives_the_same_draw_and_a_different_seed_does_not() -> None:
    samples = [_sample(i, hallucinated=i % 2 == 0) for i in range(30)]

    assert [s.id for s in shuffled(samples, 7)] == [s.id for s in shuffled(samples, 7)]
    assert [s.id for s in shuffled(samples, 7)] != [s.id for s in shuffled(samples, 8)]


def test_stratified_sampling_preserves_the_label_balance() -> None:
    """An unstratified draw moves every threshold metric for reasons unrelated to the judge."""
    samples = [_sample(i, hallucinated=i < 30) for i in range(100)]  # 30% positive

    taken = stratified_sample(samples, 40)

    assert len(taken) == 40
    assert sum(s.label_hallucinated for s in taken) == 12  # 30% of 40


def test_stratified_sampling_spans_every_task_type() -> None:
    samples = [
        _sample(i, hallucinated=i % 2 == 0, task=("qa", "summary", "data2txt")[i % 3])
        for i in range(60)
    ]

    taken = stratified_sample(samples, 12)

    assert {s.meta["task_type"] for s in taken} == {"qa", "summary", "data2txt"}


def test_dataset_stats_report_balance_and_lengths() -> None:
    samples = [_sample(i, hallucinated=i < 4) for i in range(10)]

    stats = describe("synthetic", samples)

    assert (stats.total, stats.hallucinated) == (10, 4)
    assert stats.hallucinated_share == pytest.approx(0.4)
    assert stats.task_types == {"qa": 10}


# --- held-out thresholds -------------------------------------------------------------


def test_the_holdout_split_is_stable_and_judge_independent() -> None:
    """Every judge must be scored on the same reporting half, or they are not comparable."""
    folds = [holdout_fold("ragtruth", f"s{i}") for i in range(200)]

    assert set(folds) == {0, 1}
    assert folds == [holdout_fold("ragtruth", f"s{i}") for i in range(200)]
    # Roughly balanced rather than degenerate.
    assert 60 < sum(folds) < 140


def test_a_holdout_threshold_is_measured_on_rows_it_never_saw() -> None:
    labels = [True, False] * 20
    scores = [0.1 if label else 0.9 for label in labels]
    # Takes indices 0,1,4,5,8,9,… so both halves hold both classes. A `% 2` mask would put
    # every positive in one half, which the guard correctly refuses to score.
    selection = [index % 4 < 2 for index in range(40)]

    result = select_threshold_holdout(labels, scores, selection)

    assert result is not None
    assert result.selection_size == 20
    assert result.reporting_size == 20
    assert result.metrics.f1 == pytest.approx(1.0)


def test_no_holdout_is_reported_when_a_half_holds_only_one_class() -> None:
    """F1 0.00 at threshold 0.00 reads as "detects nothing"; the truth is "cannot measure"."""
    labels = [False] * 10
    scores = [0.5] * 10

    assert select_threshold_holdout(labels, scores, [i < 5 for i in range(10)]) is None


# --- kappa against gold --------------------------------------------------------------


def test_kappa_against_gold_is_one_for_a_perfect_judge() -> None:
    metrics = threshold_metrics([True, True, False, False], [0.0, 0.1, 0.9, 1.0], 0.5)

    assert metrics.cohen_kappa == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(1.0)


def test_kappa_is_near_zero_for_a_judge_that_only_matches_by_chance() -> None:
    """Accuracy alone would flatter this judge; kappa is what exposes it."""
    labels = [True, False, True, False]
    scores = [0.0, 0.0, 1.0, 1.0]  # right half the time, by construction

    metrics = threshold_metrics(labels, scores, 0.5)

    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.cohen_kappa == pytest.approx(0.0)


# --- privacy ablation ----------------------------------------------------------------


def _rows_file(path: Path, rows: list[dict[str, Any]], privacy_mode: str, run_id: str) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path.with_name(f"{path.stem}.manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "failure_count": 0,
                "failures": [],
                "run_identity": {"privacy_mode": privacy_mode},
            }
        ),
        encoding="utf-8",
    )
    return path


def _scored(index: int, score: float, hallucinated: bool) -> dict[str, Any]:
    return {
        "sample_id": f"s{index}",
        "dataset": "d",
        "judge": "slm",
        "model": "m",
        "label_hallucinated": hallucinated,
        "scores": {"faithfulness": score},
        "verdicts": [],
        "latency_ms": 10.0,
        "usage": {"total_tokens": 10},
    }


def test_the_privacy_ablation_pairs_the_masked_and_unmasked_runs(tmp_path: Path) -> None:
    unmasked = [_scored(i, 0.1 if i < 6 else 0.9, i < 6) for i in range(12)]
    # Masking costs this judge two samples: it now scores two real hallucinations as
    # faithful, which is the direction the ablation exists to detect.
    masked = [
        _scored(i, 0.9 if i in (0, 1) else unmasked[i]["scores"]["faithfulness"], i < 6)
        for i in range(12)
    ]
    files = [
        _rows_file(tmp_path / "d_slm.jsonl", masked, "mask", "run-mask"),
        _rows_file(tmp_path / "d_slm_off.jsonl", unmasked, "off", "run-off"),
    ]
    frame = assign_series(load_rows(files))
    reports = {
        str(series): analyze_judge(frame[frame["series"] == series])
        for series in sorted(frame["series"].dropna().unique())
    }

    ablations = analyze_privacy_ablation(frame, reports)

    assert len(ablations) == 1
    ablation = ablations[0]
    assert ablation.shared_samples == 12
    assert ablation.delta_f1 is not None
    assert ablation.delta_f1 < 0, "masking mislabeled two samples, so F1 must drop"


def test_no_ablation_is_invented_when_only_one_privacy_mode_was_run(tmp_path: Path) -> None:
    rows = [_scored(i, 0.5, i < 3) for i in range(6)]
    frame = assign_series(load_rows([_rows_file(tmp_path / "d_slm.jsonl", rows, "mask", "r")]))
    reports = {"slm": analyze_judge(frame)}

    assert analyze_privacy_ablation(frame, reports) == []


# --- error analysis ------------------------------------------------------------------


def test_buckets_describe_the_sample_not_the_judgement() -> None:
    buckets = sample_buckets(contexts_chars=5000, answer="It carried 3 units and launched.",
                             context_count=3)

    assert set(buckets) >= {"long_context", "multi_passage", "numeric_claim", "conjoined_claims"}


def test_the_error_rate_denominator_counts_sample_judge_pairs() -> None:
    """Four judges over one sample give four chances to be wrong, not one."""
    rows = [
        {
            "sample_id": "s1",
            "dataset": "d",
            "series": name,
            "faithfulness": 0.5,
            "label_hallucinated": True,
            "buckets": ("numeric_claim",),
        }
        for name in ("a", "b", "c", "d")
    ]

    assert bucket_totals(rows) == {"numeric_claim": 4}


def test_unscored_rows_are_excluded_from_the_error_denominator() -> None:
    rows = [
        {"sample_id": "s1", "dataset": "d", "series": "a", "faithfulness": None,
         "label_hallucinated": True, "buckets": ("plain",)},
        {"sample_id": "s2", "dataset": "d", "series": "a", "faithfulness": 0.4,
         "label_hallucinated": True, "buckets": ("plain",)},
    ]

    assert bucket_totals(rows) == {"plain": 1}


def test_only_disagreements_with_the_human_label_become_error_cases() -> None:
    from slm_rag_eval.bench.analyze import HoldoutResult

    report = analyze_judge(
        load_rows(
            [
                _rows_file(
                    Path(__file__).parent / "_tmp_errors.jsonl",
                    [_scored(i, 0.1 if i < 2 else 0.9, i < 2) for i in range(4)],
                    "mask",
                    "r",
                )
            ]
        )
    )
    report.holdout = HoldoutResult(0.5, 2, 2, threshold_metrics([True], [0.1], 0.5))
    rows = [
        {"sample_id": "s0", "dataset": "d", "series": "slm", "faithfulness": 0.1,
         "label_hallucinated": True, "buckets": ("plain",)},   # correct
        {"sample_id": "s2", "dataset": "d", "series": "slm", "faithfulness": 0.9,
         "label_hallucinated": True, "buckets": ("plain",)},   # missed hallucination
    ]

    cases = collect_errors(rows, {"slm": report})

    assert [case.sample_id for case in cases] == ["s2"]
    assert cases[0].kind == "false_negative"
    (Path(__file__).parent / "_tmp_errors.jsonl").unlink(missing_ok=True)
    (Path(__file__).parent / "_tmp_errors.manifest.json").unlink(missing_ok=True)


# --- environment and orchestration ---------------------------------------------------


def test_the_environment_table_records_what_a_reader_needs() -> None:
    captured = environment_module.capture()
    table = environment_module.render(captured)

    assert captured.python_version
    assert "### Experimental environment" in table
    assert "| Python |" in table
    if not captured.gpus:
        assert "CPU inference" in table, "a CPU-only run must say so beside its latencies"


def test_the_experiment_plan_covers_every_judge_and_the_ablation() -> None:
    legs = plan_legs(["a", "b"], cloud=True, ablation_model="a")

    assert [leg.name for leg in legs] == ["slm-a", "slm-b", "slm-a-nomask", "cloud"]
    assert [leg.privacy_mode for leg in legs] == ["mask", "mask", "off", "mask"]


def test_the_plan_can_omit_the_cloud_baseline_and_the_ablation() -> None:
    legs = plan_legs(["a"], cloud=False, ablation_model=None)

    assert [leg.name for leg in legs] == ["slm-a"]


def test_analyze_writes_the_machine_readable_results_file(tmp_path: Path) -> None:
    """The report quotes exact numbers; re-typing them from Markdown is how they drift."""
    rows = [_scored(i, 0.1 if i < 5 else 0.9, i < 5) for i in range(10)]
    outcome = analyze([_rows_file(tmp_path / "d_slm.jsonl", rows, "mask", "r")], tmp_path / "out")

    payload = json.loads(Path(outcome["results_path"]).read_text())
    assert "judges" in payload
    entry = next(iter(payload["judges"].values()))
    assert entry["coverage"]["scored"] == 10
    assert entry["held_out"] is not None
    assert "cohen_kappa" in entry["held_out"]
