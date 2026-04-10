from __future__ import annotations

from pathlib import Path

import pytest

from slm_rag_eval.bench.download import DATASET_SOURCES, download_dataset


def test_download_writes_each_file_once_and_reuses_the_cache(tmp_path: Path) -> None:
    fetched: list[tuple[str, Path]] = []

    def fake_fetch(url: str, destination: Path) -> None:
        fetched.append((url, destination))
        destination.write_text("{}\n", encoding="utf-8")

    paths = download_dataset("ragtruth", data_dir=tmp_path, fetch=fake_fetch)

    assert {path.name for path in paths} == {"response.jsonl", "source_info.jsonl"}
    assert all(path.parent == tmp_path / "ragtruth" for path in paths)
    assert len(fetched) == 2

    # A second run is a no-op: cached files are not re-downloaded.
    download_dataset("ragtruth", data_dir=tmp_path, fetch=fake_fetch)
    assert len(fetched) == 2

    download_dataset("ragtruth", data_dir=tmp_path, force=True, fetch=fake_fetch)
    assert len(fetched) == 4


def test_sources_point_at_the_official_repositories() -> None:
    assert all(
        url.startswith("https://raw.githubusercontent.com/ParticleMedia/RAGTruth/")
        for url in DATASET_SOURCES["ragtruth"].values()
    )
    assert all(
        url.startswith("https://raw.githubusercontent.com/RUCAIBox/HaluEval/")
        for url in DATASET_SOURCES["halueval"].values()
    )


def test_unknown_dataset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Unknown dataset: nope.*halueval, ragtruth"):
        download_dataset("nope", data_dir=tmp_path)
