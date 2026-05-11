from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from slm_rag_eval.bench.reproduce import (
    OfflineStubJudge,
    judge_is_reachable,
    reproduce,
    reproduction_samples,
)
from slm_rag_eval.core.config import Settings

FIXTURE_DATA = Path(__file__).resolve().parents[1] / "data"


async def test_stub_returns_one_uncertain_verdict_per_requested_claim() -> None:
    judge = OfflineStubJudge()
    claims = [{"index": 0, "claim": "First claim."}, {"index": 1, "claim": "Second claim."}]

    response = await judge.complete(
        [{"role": "user", "content": "Claims to verify:\n" + json.dumps(claims)}]
    )

    verdicts = json.loads(response.text)
    assert [verdict["claim"] for verdict in verdicts] == ["First claim.", "Second claim."]
    # A stub must never claim to have verified anything.
    assert {verdict["verdict"] for verdict in verdicts} == {"uncertain"}
    assert response.model == "offline-stub"


async def test_stub_extracts_claims_when_no_verification_batch_is_present() -> None:
    judge = OfflineStubJudge()

    response = await judge.complete(
        [{"role": "user", "content": "Answer:\nThe probe launched. It carried instruments."}]
    )

    assert json.loads(response.text)["claims"] == [
        "The probe launched.",
        "It carried instruments.",
    ]


def test_unreachable_backend_falls_back_to_the_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("no judge here")

    monkeypatch.setattr(httpx, "get", refuse)

    assert judge_is_reachable(Settings(_env_file=None)) is False


def test_reachable_backend_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "get", lambda *args, **kwargs: httpx.Response(200, request=httpx.Request("GET", "/"))
    )

    assert judge_is_reachable(Settings(_env_file=None)) is True


def test_samples_prefer_a_downloaded_dataset_then_fall_back(tmp_path: Path) -> None:
    from_fixture = reproduction_samples(Settings(_env_file=None), FIXTURE_DATA)
    assert from_fixture[0].meta["dataset"] == "ragtruth"

    builtin = reproduction_samples(Settings(_env_file=None), tmp_path)
    assert builtin
    assert all(sample.meta["dataset"] == "builtin-synthetic" for sample in builtin)
    # The built-in set must contain both classes or the report would be meaningless.
    assert {sample.label_hallucinated for sample in builtin} == {True, False}


def test_reproduce_writes_rows_and_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLMEVAL_PRIVACY_MODE", "off")
    monkeypatch.setattr(
        "slm_rag_eval.bench.reproduce.judge_is_reachable", lambda settings, **kwargs: False
    )

    manifest = reproduce(out_dir=tmp_path, data_dir=tmp_path / "no-datasets-here")

    assert manifest["judge"] == "stub"
    assert manifest["model"] == "offline-stub"
    assert manifest["failure_count"] == 0
    assert manifest["samples_written"] == manifest["samples_seen"]
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "roc_curves.png").exists()
    assert (tmp_path / "rows" / "reproduce_stub.jsonl").exists()


def test_builtin_fallback_provides_the_documented_run_size() -> None:
    """`make reproduce` claims a 20-sample bench; the offline path must deliver one."""
    from slm_rag_eval.bench.reproduce import SAMPLE_COUNT, _builtin_samples

    samples = _builtin_samples()

    assert len(samples) == SAMPLE_COUNT == 20
    assert len({sample.id for sample in samples}) == SAMPLE_COUNT
    # Balanced labels, or every threshold metric would be degenerate.
    assert sum(sample.label_hallucinated for sample in samples) == SAMPLE_COUNT // 2
    assert all(sample.contexts for sample in samples)


def test_reproduce_scores_twenty_samples_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLMEVAL_PRIVACY_MODE", "off")
    monkeypatch.setattr(
        "slm_rag_eval.bench.reproduce.judge_is_reachable", lambda settings, **kwargs: False
    )

    manifest = reproduce(out_dir=tmp_path, data_dir=tmp_path / "no-datasets-here")

    assert manifest["samples_seen"] == 20
    assert manifest["samples_written"] == 20
    assert manifest["failure_count"] == 0
