"""Pure helpers shared by the CLI and the Streamlit dashboard.

Everything here is deterministic and free of I/O and framework imports, so the dashboard can
stay a thin rendering layer and all of its logic can be unit-tested.
"""

from __future__ import annotations

from typing import Any

from slm_rag_eval.core.config import Settings
from slm_rag_eval.core.schemas import EvalRequest, EvalResult
from slm_rag_eval.metrics.registry import RequestSanitizer, sanitize_for_judge


def sanitized_preview(
    request: EvalRequest,
    *,
    settings: Settings,
    sanitizer: RequestSanitizer | None = None,
) -> tuple[EvalRequest, dict[str, str]]:
    """What the judge will actually see, plus the (in-memory only) placeholder mapping."""
    return sanitize_for_judge(request, settings=settings, sanitizer=sanitizer)


def highlight_placeholders(text: str, mapping: dict[str, str], *, marker: str = "**") -> str:
    """Wrap every placeholder occurrence in `marker` so masked spans stand out in Markdown."""
    if not mapping:
        return text
    highlighted = text
    for placeholder in sorted(mapping, key=len, reverse=True):
        highlighted = highlighted.replace(placeholder, f"{marker}{placeholder}{marker}")
    return highlighted


def verdict_rows(result: EvalResult) -> list[dict[str, str]]:
    """Per-claim table rows: claim, verdict, reason."""
    return [
        {"claim": verdict.claim, "verdict": verdict.verdict, "reason": verdict.reason}
        for verdict in result.verdicts
    ]


def score_summary(result: EvalResult) -> dict[str, float | None]:
    """The scores a dashboard shows as metrics; `None` means the judge could not score it."""
    return {"faithfulness": result.faithfulness, "relevance": result.relevance}


def total_latency_ms(result: EvalResult) -> float:
    """Sum of the per-stage timings recorded on the result."""
    return float(sum(result.timings.values()))


def token_totals(result: EvalResult) -> dict[str, int]:
    """Token usage summed across metrics, whether or not the registry namespaced it."""
    totals: dict[str, int] = {}
    for value in result.model_info.values():
        if not isinstance(value, dict):
            continue
        usage = value.get("token_usage")
        if not isinstance(usage, dict):
            continue
        for key, count in usage.items():
            if isinstance(count, int):
                totals[key] = totals.get(key, 0) + count
    return totals


def format_verdict_table(rows: list[dict[str, str]], *, width: int = 60) -> str:
    """Fixed-width claim table for the terminal; deterministic so it can be captured in docs."""
    if not rows:
        return "(no claims extracted)"

    claim_width = max(len("CLAIM"), *(len(row["claim"]) for row in rows))
    claim_width = min(claim_width, width)
    verdict_width = max(len("VERDICT"), *(len(row["verdict"]) for row in rows))

    lines = [f"{'CLAIM':<{claim_width}}  {'VERDICT':<{verdict_width}}  REASON"]
    lines.append("-" * (claim_width + verdict_width + len("REASON") + 4))
    for row in rows:
        claim = _truncate(row["claim"], claim_width)
        lines.append(f"{claim:<{claim_width}}  {row['verdict']:<{verdict_width}}  {row['reason']}")
    return "\n".join(lines)


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def batch_row(
    sample_id: str,
    result: EvalResult,
    *,
    model: str,
    latency_ms: float,
    dataset: str = "cli",
    judge: str = "cli",
) -> dict[str, Any]:
    """One `rageval batch` output row: the bench.run schema minus the human label.

    `dataset` and `judge` are part of that schema, not decoration: the analysis groups
    series by judge and keys samples by (dataset, sample_id), so rows without them cannot
    be analyzed at all.
    """
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "judge": judge,
        "model": model,
        "scores": score_summary(result),
        "verdicts": [verdict.model_dump() for verdict in result.verdicts],
        "latency_ms": latency_ms,
        "usage": token_totals(result),
    }
