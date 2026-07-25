"""The rental-day machinery has to be right the first time, on a machine that then vanishes.

Covers the pieces with no second chance: the dataset lock that proves two hosts hold the
same data, the bundle that decides what gets copied off, and the model freeze that turns a
mutable tag into an identifiable artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from slm_rag_eval.bench.bundle import EXPECTED, build, sha256_of
from slm_rag_eval.bench.campaign import _best_model, _child_env
from slm_rag_eval.bench.dataset_lock import build_lock, digest_file, read_lock, verify_lock
from slm_rag_eval.bench.models import ModelArtifact, fetch_artifact, render
from slm_rag_eval.bench.resources import ResourceSampler, ResourceUsage

FIXTURE_DATA = Path(__file__).resolve().parents[1] / "data"


# --- dataset lock --------------------------------------------------------------------


def test_a_lock_matches_the_data_it_was_built_from() -> None:
    lock = build_lock("halueval", data_dir=FIXTURE_DATA, limit=2)

    assert verify_lock(lock, data_dir=FIXTURE_DATA) == []


def test_a_changed_file_is_reported_as_a_digest_mismatch(tmp_path: Path) -> None:
    """The failure this exists for: two hosts fetched a mutable branch at different times."""
    source = FIXTURE_DATA / "halueval" / "qa_data.json"
    copied = tmp_path / "halueval"
    copied.mkdir()
    (copied / "qa_data.json").write_text(source.read_text(), encoding="utf-8")
    lock = build_lock("halueval", data_dir=tmp_path, limit=2)

    (copied / "qa_data.json").write_text(
        source.read_text() + json.dumps({"question": "extra", "knowledge": "k",
                                         "right_answer": "a", "hallucinated_answer": "b"}) + "\n",
        encoding="utf-8",
    )
    problems = verify_lock(lock, data_dir=tmp_path)

    assert any("sha256" in problem for problem in problems)


def test_a_missing_file_is_reported_rather_than_passing_quietly(tmp_path: Path) -> None:
    lock = build_lock("halueval", data_dir=FIXTURE_DATA, limit=2)

    problems = verify_lock(lock, data_dir=tmp_path)

    assert problems and "missing" in problems[0]


def test_the_lock_records_the_drawn_identities_not_just_the_count() -> None:
    """A different draw of the same size must not verify: the samples decide the numbers."""
    first = build_lock("halueval", data_dir=FIXTURE_DATA, limit=2, seed=1)
    second = build_lock("halueval", data_dir=FIXTURE_DATA, limit=2, seed=999)

    assert first.drawn_samples == second.drawn_samples
    assert first.drawn_sample_ids_sha256 != second.drawn_sample_ids_sha256


def test_a_lock_survives_a_round_trip_through_json(tmp_path: Path) -> None:
    from dataclasses import asdict

    lock = build_lock("halueval", data_dir=FIXTURE_DATA, limit=2)
    path = tmp_path / "dataset.lock.json"
    path.write_text(json.dumps(asdict(lock)), encoding="utf-8")

    assert verify_lock(read_lock(path), data_dir=FIXTURE_DATA) == []


def test_digest_reports_size_and_line_count(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    digest = digest_file(path)

    assert (digest.bytes, digest.lines) == (6, 3)
    assert digest.sha256 == sha256_of(path)


# --- bundle ---------------------------------------------------------------------------


def _artifact_tree(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")


def test_a_complete_bundle_reports_nothing_required_missing(tmp_path: Path) -> None:
    _artifact_tree(tmp_path, [item.relative_path for item in EXPECTED])

    missing, archive = build(tmp_path, archive=False)

    assert [item for item in missing if item.required] == []
    assert archive is None


def test_a_missing_required_artifact_is_listed_rather_than_omitted(tmp_path: Path) -> None:
    """A gap the reader can see is recoverable; one they cannot see is not."""
    _artifact_tree(tmp_path, [item.relative_path for item in EXPECTED if item.required][:3])

    build(tmp_path, archive=False)
    index = (tmp_path / "INDEX.md").read_text()

    assert "**NO — required**" in index
    assert "summary.md" in index


def test_the_index_maps_every_artifact_to_a_report_item(tmp_path: Path) -> None:
    _artifact_tree(tmp_path, ["summary.md", "results.json"])

    build(tmp_path, archive=False)
    index = (tmp_path / "INDEX.md").read_text()

    assert "Tables 6.1–6.3" in index
    assert "held-out" in index, "the index must say which numbers to quote"


def test_the_manifest_covers_every_file_and_verifies(tmp_path: Path) -> None:
    _artifact_tree(tmp_path, ["summary.md", "errors/error_analysis.md"])

    build(tmp_path, archive=False)
    entries = (tmp_path / "MANIFEST.sha256").read_text().splitlines()

    assert len(entries) == 3, "two artifacts plus the INDEX.md written beside them"
    for entry in entries:
        digest, name = entry.split("  ", 1)
        assert sha256_of(tmp_path / name) == digest


def test_the_archive_is_written_beside_the_directory(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    _artifact_tree(root, ["summary.md"])

    _, archive = build(root, archive=True)

    assert archive is not None and archive.exists()
    assert archive.name == "experiment.tar.gz"


def test_rows_are_listed_because_they_are_the_irreplaceable_part(tmp_path: Path) -> None:
    _artifact_tree(tmp_path, ["rows/slm/ragtruth_slm.jsonl", "summary.md"])

    build(tmp_path, archive=False)

    assert "rows/slm/ragtruth_slm.jsonl" in (tmp_path / "INDEX.md").read_text()


# --- model freeze ---------------------------------------------------------------------


def test_model_details_come_from_the_running_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
                "model_info": {"qwen3.context_length": 40960},
                "license": "Apache License 2.0\nmore text",
            },
            request=httpx.Request("POST", url),
        )

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:8b", "digest": "sha256:abc123", "size": 5_200_000_000}
                ]
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    artifact = fetch_artifact("qwen3:8b", "http://localhost:11434/v1")

    assert artifact.parameter_size == "8.2B"
    assert artifact.quantization == "Q4_K_M"
    assert artifact.context_length == 40960
    assert artifact.digest == "sha256:abc123"
    assert artifact.license_name == "Apache License 2.0"


def test_an_unreachable_server_leaves_fields_empty_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)

    artifact = fetch_artifact("qwen3:8b", "http://localhost:11434/v1")

    assert artifact.digest is None
    assert artifact.quantization is None


def test_the_model_table_says_why_a_digest_matters() -> None:
    artifact = ModelArtifact(
        tag="qwen3:8b", digest="sha256:abcdef0123456789", size_bytes=5_000_000_000
    )
    table = render([artifact])

    assert "qwen3:8b" in table
    assert "republishable" in table


# --- campaign helpers -----------------------------------------------------------------


def test_the_ablation_subject_is_the_best_held_out_f1(tmp_path: Path) -> None:
    """Chosen from recorded results, not by eye, so the choice is auditable."""
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "judges": {
                    "slm:a": {"model": "a", "held_out": {"f1": 0.61}},
                    "slm:b": {"model": "b", "held_out": {"f1": 0.83}},
                    "slm:c": {"model": "c", "held_out": None},
                }
            }
        ),
        encoding="utf-8",
    )

    assert _best_model(results, ["a", "b", "c"]) == "b"


def test_no_ablation_subject_when_nothing_has_a_held_out_score(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"judges": {"slm:a": {"model": "a", "held_out": None}}}), "utf-8")

    assert _best_model(results, ["a"]) is None


def test_a_missing_results_file_is_not_fatal(tmp_path: Path) -> None:
    assert _best_model(tmp_path / "absent.json", ["a"]) is None


def test_the_child_environment_puts_this_interpreter_first_on_path() -> None:
    """Otherwise `make check` runs whichever pytest the shell happens to expose."""
    import sys

    env = _child_env({"SLMEVAL_MODEL": "x"})

    assert env["PATH"].startswith(str(Path(sys.executable).parent))
    assert env["SLMEVAL_MODEL"] == "x"


# --- resources ------------------------------------------------------------------------


def test_the_sampler_records_at_least_one_reading() -> None:
    with ResourceSampler(interval_s=0.05) as sampler:
        pass

    assert sampler.usage.samples >= 2, "one on entry and one on exit at minimum"
    assert sampler.usage.duration_s >= 0


def test_usage_renders_readably_without_a_gpu() -> None:
    assert "no GPU detected" in ResourceUsage().render()
