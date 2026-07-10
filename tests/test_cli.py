from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slm_rag_eval import cli
from slm_rag_eval.core.config import Settings
from tests.conftest import FakeLLMClient
from tests.support import verdict_batch

runner = CliRunner()

CLAIM = "The module uses a ceramic shield."


def _script(judge: FakeLLMClient, verdict: str = "supported") -> FakeLLMClient:
    judge.push(
        json.dumps({"claims": [CLAIM]}),
        verdict_batch(CLAIM, verdict=verdict),
    )
    return judge


@pytest.fixture
def scripted_cli(monkeypatch: pytest.MonkeyPatch) -> FakeLLMClient:
    """Inject a scripted judge and offline settings through the CLI's DI seam."""
    judge = FakeLLMClient()
    monkeypatch.setattr(cli, "judge_factory", lambda settings: judge)
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(_env_file=None, privacy_mode="off", model="synthetic-slm"),
    )
    return judge


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(cli.app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip()


def test_eval_prints_the_claim_table_and_scores(scripted_cli: FakeLLMClient) -> None:
    _script(scripted_cli)

    result = runner.invoke(
        cli.app,
        [
            "eval",
            "--question",
            "Which material shields the module?",
            "--answer",
            CLAIM,
            "--context",
            "The module is protected by a ceramic shield.",
        ],
    )

    assert result.exit_code == 0
    assert "CLAIM" in result.stdout
    assert "supported" in result.stdout
    assert "faithfulness: 1.000" in result.stdout
    assert "judge: synthetic-slm" in result.stdout
    assert "privacy: off" in result.stdout


def test_eval_reads_a_json_request_file(scripted_cli: FakeLLMClient, tmp_path: Path) -> None:
    _script(scripted_cli)
    request_file = tmp_path / "request.json"
    request_file.write_text(
        json.dumps(
            {
                "question": "Which material shields the module?",
                "answer": CLAIM,
                "contexts": ["The module is protected by a ceramic shield."],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["eval", "--json", str(request_file)])

    assert result.exit_code == 0
    assert "faithfulness: 1.000" in result.stdout


def test_eval_without_question_or_json_is_a_usage_error(scripted_cli: FakeLLMClient) -> None:
    result = runner.invoke(cli.app, ["eval", "--answer", CLAIM])

    assert result.exit_code != 0
    assert "--json" in result.output


def test_fail_under_sets_the_exit_code(scripted_cli: FakeLLMClient) -> None:
    _script(scripted_cli, verdict="unsupported")

    result = runner.invoke(
        cli.app,
        [
            "eval",
            "--question",
            "Which material shields the module?",
            "--answer",
            CLAIM,
            "--fail-under",
            "0.5",
        ],
    )

    assert result.exit_code == 1
    assert "FAIL: faithfulness 0.000 < --fail-under 0.5" in result.stdout


def test_fail_under_defaults_to_never_failing(scripted_cli: FakeLLMClient) -> None:
    _script(scripted_cli, verdict="unsupported")

    result = runner.invoke(
        cli.app, ["eval", "--question", "Q?", "--answer", CLAIM]
    )

    assert result.exit_code == 0
    assert "faithfulness: 0.000" in result.stdout


def test_batch_writes_one_row_per_input_line(
    scripted_cli: FakeLLMClient, tmp_path: Path
) -> None:
    _script(scripted_cli)
    _script(scripted_cli)
    input_file = tmp_path / "batch.jsonl"
    input_file.write_text(
        "\n".join(
            json.dumps({"id": name, "question": "Q?", "answer": CLAIM, "contexts": ["ctx"]})
            for name in ("a", "b")
        )
        + "\n",
        encoding="utf-8",
    )
    out_file = tmp_path / "out" / "results.jsonl"

    result = runner.invoke(cli.app, ["batch", str(input_file), "--out", str(out_file)])

    assert result.exit_code == 0
    rows = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert set(rows[0]) == {
        "sample_id",
        "dataset",
        "judge",
        "model",
        "scores",
        "verdicts",
        "latency_ms",
        "usage",
    }
    assert rows[0]["dataset"] == "batch"  # the input file stem
    assert rows[0]["judge"] == "cli"
    # The bench schema minus the human label, which a CLI batch has no way to know.
    assert "label_hallucinated" not in rows[0]
    assert rows[0]["scores"]["faithfulness"] == 1.0


def test_demo_transcript_matches_the_documented_output(
    scripted_cli: FakeLLMClient,
) -> None:
    """docs/demo.md quotes this transcript; regenerate it there if the CLI output changes."""
    claims = [
        "The Aurora probe carried 4 instruments.",
        "The Aurora probe launched in 2019.",
    ]
    scripted_cli.push(
        json.dumps({"claims": claims}),
        json.dumps(
            {
                "verdicts": [
                    {
                        "claim": claims[0],
                        "verdict": "unsupported",
                        "reason": "The context says 3 instruments.",
                    },
                    {
                        "claim": claims[1],
                        "verdict": "supported",
                        "reason": "The context gives the same year.",
                    },
                ]
            }
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "eval",
            "--question",
            "What did the Aurora probe carry?",
            "--answer",
            "The Aurora probe carried 4 instruments and launched in 2019.",
            "--context",
            "Aurora launched in 2019 carrying 3 instruments.",
        ],
    )

    assert result.exit_code == 0
    # Timings are the only varying part, so they are normalized before comparison.
    transcript = re.sub(r"\d+\.\d ms", "<ms> ms", result.stdout).replace(
        "judge: synthetic-slm", "judge: qwen2.5:7b-instruct"
    )
    documented = (Path(__file__).resolve().parents[1] / "docs" / "demo.md").read_text()
    assert transcript.strip() in documented


def test_batch_output_can_be_analyzed_by_the_bench_analysis(
    scripted_cli: FakeLLMClient, tmp_path: Path
) -> None:
    """The README promises batch rows feed the same analysis as bench.run — prove it."""
    from slm_rag_eval.bench.analyze import analyze

    for _ in range(2):
        _script(scripted_cli)
    input_file = tmp_path / "questions.jsonl"
    input_file.write_text(
        "\n".join(
            json.dumps({"id": name, "question": "Q?", "answer": CLAIM, "contexts": ["ctx"]})
            for name in ("a", "b")
        )
        + "\n",
        encoding="utf-8",
    )
    out_file = tmp_path / "rows.jsonl"

    assert runner.invoke(cli.app, ["batch", str(input_file), "--out", str(out_file)]).exit_code == 0

    outcome = analyze([out_file], tmp_path / "report")

    # Grouping needs `judge`; keying samples needs `dataset`. Missing either used to make
    # this call produce an empty report or fail outright.
    assert list(outcome["reports"]) == ["cli"]
    report = outcome["reports"]["cli"]
    assert report.scored == 2
    assert report.datasets == ("questions",)
    assert (tmp_path / "report" / "summary.md").exists()
