from __future__ import annotations

from pathlib import Path

import pytest

from slm_rag_eval.bench.datasets import (
    DatasetNotDownloadedError,
    available_datasets,
    load,
)

FIXTURE_DATA = Path(__file__).resolve().parents[1] / "data"


def test_ragtruth_loader_joins_sources_and_maps_span_labels() -> None:
    samples = load("ragtruth", data_dir=FIXTURE_DATA)

    # The row pointing at a missing source_id is dropped rather than half-loaded.
    assert [sample.id for sample in samples] == ["r1", "r2", "r3", "r4"]

    supported, hallucinated = samples[0], samples[1]
    assert supported.label_hallucinated is False
    assert hallucinated.label_hallucinated is True
    assert hallucinated.meta["label_count"] == 1

    # QA: the question comes from source_info and passages are split into separate contexts.
    assert supported.question == "Which alloy shields the probe?"
    assert supported.contexts == [
        "The probe is shielded by a titanium alloy.",
        "The alloy was chosen for its heat tolerance.",
    ]
    assert supported.answer == "The probe is shielded by a titanium alloy."
    assert supported.meta["task_type"] == "QA"
    assert supported.meta["response_model"] == "synthetic-judge-a"


def test_ragtruth_loader_normalizes_summary_and_structured_tasks() -> None:
    samples = {sample.id: sample for sample in load("ragtruth", data_dir=FIXTURE_DATA)}

    summary = samples["r3"]
    assert summary.question == "Summarize the article."
    assert summary.contexts == [
        "The observatory recorded three tremors on Tuesday. No damage was reported."
    ]

    structured = samples["r4"]
    assert structured.question == "Describe the venue."
    assert len(structured.contexts) == 1
    assert "Blue Harbour Cafe" in structured.contexts[0]


def test_halueval_loader_yields_one_positive_and_one_negative_per_row() -> None:
    samples = load("halueval", data_dir=FIXTURE_DATA)

    assert len(samples) == 10
    assert [sample.id for sample in samples[:2]] == ["0-right", "0-hallucinated"]
    assert [sample.label_hallucinated for sample in samples[:2]] == [False, True]
    assert samples[0].question == "When did the Northport observatory open?"
    assert samples[0].contexts == [
        "The Northport observatory opened in 1971. It houses a single refracting telescope."
    ]
    assert sum(sample.label_hallucinated for sample in samples) == 5


def test_limit_truncates_and_must_be_positive() -> None:
    assert len(load("halueval", 3, data_dir=FIXTURE_DATA)) == 3

    with pytest.raises(ValueError, match="limit must be positive"):
        load("halueval", 0, data_dir=FIXTURE_DATA)


def test_unknown_dataset_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match=r"Unknown dataset: nope.*halueval, ragtruth"):
        load("nope", data_dir=FIXTURE_DATA)

    assert available_datasets() == ("halueval", "ragtruth")


def test_missing_cache_names_the_download_script(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotDownloadedError) as excinfo:
        load("ragtruth", data_dir=tmp_path)

    assert "scripts/download_ragtruth.py" in str(excinfo.value)
    assert excinfo.value.dataset == "ragtruth"
