"""Unit tests for the pure functions the dashboard renders (never Streamlit itself)."""

from __future__ import annotations

from slm_rag_eval.core.config import Settings
from slm_rag_eval.core.schemas import ClaimVerdict, EvalRequest, EvalResult
from slm_rag_eval.demo import (
    batch_row,
    format_verdict_table,
    highlight_placeholders,
    sanitized_preview,
    score_summary,
    token_totals,
    total_latency_ms,
    verdict_rows,
)
from slm_rag_eval.privacy.sanitizer import Sanitizer
from tests.privacy.test_sanitizer import StubAnalyzer

# Invented data only (AGENTS.md rule 3).
NAME = "Rowan Quill"


def _result() -> EvalResult:
    return EvalResult(
        faithfulness=0.5,
        relevance=None,
        verdicts=[
            ClaimVerdict(claim="A short claim.", verdict="supported", reason="Context agrees."),
            ClaimVerdict(
                claim="A very long claim that runs past the table width limit for sure.",
                verdict="unsupported",
                reason="Context disagrees.",
            ),
        ],
        model_info={
            "faithfulness": {"token_usage": {"prompt_tokens": 10, "completion_tokens": 4}},
            "relevance": {"token_usage": {"prompt_tokens": 6, "completion_tokens": 2}},
        },
        timings={"faithfulness_ms": 12.5, "relevance_ms": 7.5},
    )


def test_sanitized_preview_returns_what_the_judge_would_see() -> None:
    request = EvalRequest(question=f"Who is {NAME}?", answer=f"{NAME} filed it.", contexts=[])
    sanitizer = Sanitizer(analyzer=StubAnalyzer({NAME: "PERSON"}))

    preview, mapping = sanitized_preview(
        request,
        settings=Settings(_env_file=None, privacy_mode="mask"),
        sanitizer=sanitizer,
    )

    assert preview.question == "Who is <PERSON_1>?"
    assert preview.answer == "<PERSON_1> filed it."
    assert mapping == {"<PERSON_1>": NAME}


def test_sanitized_preview_is_a_passthrough_when_privacy_is_off() -> None:
    request = EvalRequest(question=f"Who is {NAME}?", answer="Someone.", contexts=[])

    preview, mapping = sanitized_preview(
        request, settings=Settings(_env_file=None, privacy_mode="off")
    )

    assert preview == request
    assert mapping == {}


def test_highlight_placeholders_marks_every_masked_span() -> None:
    text = "<PERSON_1> emailed <EMAIL_ADDRESS_1> about <PERSON_1>."
    mapping = {"<PERSON_1>": NAME, "<EMAIL_ADDRESS_1>": "rowan@example.com"}

    assert highlight_placeholders(text, mapping) == (
        "**<PERSON_1>** emailed **<EMAIL_ADDRESS_1>** about **<PERSON_1>**."
    )
    assert highlight_placeholders("nothing masked", {}) == "nothing masked"


def test_verdict_rows_and_score_summary_flatten_the_result() -> None:
    result = _result()

    assert verdict_rows(result) == [
        {"claim": "A short claim.", "verdict": "supported", "reason": "Context agrees."},
        {
            "claim": "A very long claim that runs past the table width limit for sure.",
            "verdict": "unsupported",
            "reason": "Context disagrees.",
        },
    ]
    assert score_summary(result) == {"faithfulness": 0.5, "relevance": None}


def test_latency_and_tokens_are_summed_across_metrics() -> None:
    result = _result()

    assert total_latency_ms(result) == 20.0
    assert token_totals(result) == {"prompt_tokens": 16, "completion_tokens": 6}


def test_verdict_table_is_aligned_and_truncated() -> None:
    table = format_verdict_table(verdict_rows(_result()), width=20)
    lines = table.splitlines()

    assert lines[0].startswith("CLAIM")
    assert "VERDICT" in lines[0]
    assert set(lines[1]) == {"-"}
    assert "…" in lines[3]  # the long claim is truncated to the requested width
    assert format_verdict_table([]) == "(no claims extracted)"


def test_batch_row_matches_the_bench_schema_without_a_label() -> None:
    row = batch_row("sample-1", _result(), model="synthetic-slm", latency_ms=20.0)

    assert set(row) == {"sample_id", "model", "scores", "verdicts", "latency_ms", "usage"}
    assert row["sample_id"] == "sample-1"
    assert row["scores"] == {"faithfulness": 0.5, "relevance": None}
    assert row["verdicts"][0]["verdict"] == "supported"
    assert row["usage"] == {"prompt_tokens": 16, "completion_tokens": 6}
