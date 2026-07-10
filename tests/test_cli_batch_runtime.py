"""`rageval batch` must survive past its first row against a real client.

The batch loop used to call `asyncio.run()` once per input row while reusing a single
judge client. Each call closed the event loop that client's connection pool was bound to,
so row two against any keep-alive backend — Ollama, vLLM, an OpenAI-compatible API — died
with `RuntimeError: Event loop is closed`. Nothing caught it, because `FakeLLMClient`
holds no connection and is happy to be used from any number of loops.

These tests pin the two properties that prevent it: one loop for the whole batch, and the
client closed once at the end.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from slm_rag_eval import cli
from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm.client import LLMResponse

runner = CliRunner()

CLAIM = "The probe carried three instruments."


class LoopBoundClient:
    """Stands in for an HTTP client whose transport belongs to one event loop.

    `httpx.AsyncClient` behaves this way: its pool is created on first use and cannot be
    driven by a different loop afterwards. Reproducing that binding here is what makes the
    regression visible without a network.
    """

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.calls = 0
        self.closed = 0

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        running = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = running
        elif running is not self.loop:
            raise RuntimeError("Event loop is closed")
        self.calls += 1
        text = json.dumps({"claims": [CLAIM]})
        if "Claims to verify" in "\n".join(str(m.get("content", "")) for m in messages):
            text = json.dumps(
                {"verdicts": [{"claim": CLAIM, "verdict": "supported",
                               "reason": "The context agrees."}]}
            )
        return LLMResponse(text=text, usage={"prompt_tokens": 1, "completion_tokens": 1})

    async def aclose(self) -> None:
        self.closed += 1


def _input_file(tmp_path: Path, rows: int) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {"id": f"s{index}", "question": "How many?", "answer": CLAIM, "contexts": [CLAIM]}
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return path


def _run_batch(tmp_path: Path, rows: int, monkeypatch: Any) -> tuple[Any, LoopBoundClient]:
    client = LoopBoundClient()
    monkeypatch.setattr(cli, "judge_factory", lambda settings: client)
    monkeypatch.setattr(
        cli, "get_settings", lambda: Settings(_env_file=None, privacy_mode="off", model="stub")
    )
    out = tmp_path / "rows.jsonl"
    result = runner.invoke(
        cli.app, ["batch", str(_input_file(tmp_path, rows)), "--out", str(out)]
    )
    return result, client


def test_every_row_is_evaluated_on_one_event_loop(tmp_path: Path, monkeypatch: Any) -> None:
    result, client = _run_batch(tmp_path, 3, monkeypatch)

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["s0", "s1", "s2"]
    assert all(row["scores"]["faithfulness"] == 1.0 for row in rows)


def test_the_judge_client_is_closed_exactly_once(tmp_path: Path, monkeypatch: Any) -> None:
    _, client = _run_batch(tmp_path, 2, monkeypatch)

    assert client.closed == 1, "a batch that leaks its client leaks sockets on every run"


def test_an_empty_metric_selection_is_refused_before_any_judge_call(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An empty selection scores nothing, so it must fail rather than succeed emptily."""
    client = LoopBoundClient()
    monkeypatch.setattr(cli, "judge_factory", lambda settings: client)
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(_env_file=None, privacy_mode="off", enabled_metrics=[]),
    )

    result = runner.invoke(
        cli.app, ["batch", str(_input_file(tmp_path, 1)), "--out", str(tmp_path / "out.jsonl")]
    )

    assert result.exit_code != 0
    assert client.calls == 0
