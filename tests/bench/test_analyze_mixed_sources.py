"""Rows from several datasets or models must not be pooled under one judge name.

tests/data/bench_mixed_sources.jsonl holds, deliberately:

    judge  model    dataset  sample_id  label         faithfulness  latency_ms
    slm    model-a  alpha    1          not-halluc.   1.0            10
    slm    model-a  alpha    2          hallucinated  0.0            20
    slm    model-b  beta     1          hallucinated  0.9            30
    slm    model-b  beta     2          not-halluc.   0.1            40
    cloud  cloud-x  alpha    1          not-halluc.   0.8           200
    cloud  cloud-x  alpha    2          hallucinated  0.2           400

`slm` is perfect on alpha and exactly inverted on beta, and sample ids repeat across
datasets — so pooling would report a mediocre middle and would pair alpha rows with beta
rows in the agreement stats.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_rag_eval.bench.analyze import analyze, analyze_agreement, assign_series, load_rows

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "bench_mixed_sources.jsonl"
EXACT = 1e-9


def test_series_are_split_by_model_and_dataset() -> None:
    frame = assign_series(load_rows([FIXTURE]))

    assert sorted(frame["series"].unique()) == [
        "cloud",  # one model, one dataset: name left alone
        "slm:model-a@alpha",
        "slm:model-b@beta",
    ]


def test_each_series_is_scored_on_its_own_rows(tmp_path: Path) -> None:
    reports = analyze([FIXTURE], tmp_path)["reports"]

    alpha = reports["slm:model-a@alpha"]
    beta = reports["slm:model-b@beta"]

    # Perfect on alpha, exactly inverted on beta. Pooled, both would have read ~0.5.
    assert alpha.roc_auc == pytest.approx(1.0, abs=EXACT)
    assert beta.roc_auc == pytest.approx(0.0, abs=EXACT)
    assert alpha.model == "model-a"
    assert beta.model == "model-b"
    assert alpha.datasets == ("alpha",)
    assert beta.datasets == ("beta",)
    # Latency is per series too, not averaged across models.
    assert alpha.latency_median_ms == pytest.approx(15.0, abs=EXACT)
    assert beta.latency_median_ms == pytest.approx(35.0, abs=EXACT)


def test_agreement_pairs_samples_by_dataset_and_id(tmp_path: Path) -> None:
    outcome = analyze([FIXTURE], tmp_path)
    agreements = {(item.judge_a, item.judge_b): item for item in outcome["agreements"]}

    # cloud and slm:model-a@alpha share alpha/1 and alpha/2 — two real pairs.
    assert agreements[("cloud", "slm:model-a@alpha")].overlap == 2

    # cloud never scored beta. Keyed on sample_id alone, ids 1 and 2 would have looked
    # shared and this pair would have been reported with overlap 2 — comparing alpha
    # scores against beta scores. It must not appear at all.
    assert ("cloud", "slm:model-b@beta") not in agreements


def test_same_sample_id_in_two_datasets_is_two_samples(tmp_path: Path) -> None:
    frame = assign_series(load_rows([FIXTURE]))
    reports = analyze([FIXTURE], tmp_path)["reports"]

    pairs = analyze_agreement(frame, reports)
    overlaps = {(item.judge_a, item.judge_b): item.overlap for item in pairs}

    assert overlaps == {("cloud", "slm:model-a@alpha"): 2}
