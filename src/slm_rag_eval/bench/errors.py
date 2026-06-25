"""Error analysis: which samples the judge got wrong, and what they have in common.

The rubric asks for failure modes with concrete examples, not just an aggregate score. A
judge at F1 0.78 is only interesting once you can say *where* the missing 0.22 lives — long
contexts, numeric claims, multi-hop reasoning — because that is what a reader can act on and
what the discussion chapter connects back to the theory.

Buckets are structural properties of the sample, computed the same way for every judge:
nothing here looks at whether the judge happened to be right, so the categories cannot be
drawn around the errors after the fact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from slm_rag_eval.bench.analyze import (
    JudgeReport,
    analyze_judge,
    assign_series,
    load_rows,
)
from slm_rag_eval.bench.datasets import LabeledSample, load

app = typer.Typer(help=__doc__, add_completion=False)

LONG_CONTEXT_CHARS = 2000
"""Above this, a context is long enough that a small model's attention is a plausible cause."""

_NUMERIC = re.compile(r"\d")
_MULTI_FACT = re.compile(r"\band\b|\bwhile\b|\balthough\b|;", re.IGNORECASE)


@dataclass(frozen=True)
class ErrorCase:
    """One sample the judge classified differently from the human label."""

    sample_id: str
    dataset: str
    series: str
    label_hallucinated: bool
    faithfulness: float
    predicted_hallucinated: bool
    buckets: tuple[str, ...]

    @property
    def kind(self) -> str:
        """False positive (flagged a faithful answer) or false negative (missed a hallucination)."""
        return "false_negative" if self.label_hallucinated else "false_positive"


def sample_buckets(
    *,
    contexts_chars: int,
    answer: str,
    context_count: int,
) -> tuple[str, ...]:
    """Structural properties of a sample, assigned without reference to any judge's answer."""
    buckets: list[str] = []
    if contexts_chars >= LONG_CONTEXT_CHARS:
        buckets.append("long_context")
    if context_count > 1:
        buckets.append("multi_passage")
    if _NUMERIC.search(answer):
        buckets.append("numeric_claim")
    if _MULTI_FACT.search(answer):
        buckets.append("conjoined_claims")
    if len(answer) < 120:
        buckets.append("short_answer")
    return tuple(buckets or ("plain",))


def collect_errors(
    rows: Sequence[dict[str, Any]],
    reports: dict[str, JudgeReport],
) -> list[ErrorCase]:
    """Every disagreement with the human label, at each series' own decision threshold."""
    cases: list[ErrorCase] = []
    for row in rows:
        series = str(row.get("series") or row.get("judge"))
        report = reports.get(series)
        if report is None:
            continue
        threshold = (
            report.holdout.threshold
            if report.holdout is not None
            else (report.best.threshold if report.best is not None else None)
        )
        score = row.get("faithfulness")
        label = row.get("label_hallucinated")
        if threshold is None or score is None or label is None:
            continue

        predicted = float(score) < threshold
        if predicted == bool(label):
            continue
        cases.append(
            ErrorCase(
                sample_id=str(row["sample_id"]),
                dataset=str(row.get("dataset")),
                series=series,
                label_hallucinated=bool(label),
                faithfulness=float(score),
                predicted_hallucinated=predicted,
                buckets=tuple(row.get("buckets") or ("plain",)),
            )
        )
    return cases


def bucket_totals(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Scored sample-judge pairs per bucket — the denominator the error rate needs.

    One entry per (sample, series), not per sample: with four judges over sixty samples
    there are 240 opportunities to be wrong, and dividing four judges' errors by sixty
    samples reports a rate up to four times too high.
    """
    totals: dict[str, int] = {}
    for row in rows:
        if row.get("faithfulness") is None or row.get("label_hallucinated") is None:
            continue
        for bucket in row.get("buckets") or ("plain",):
            totals[bucket] = totals.get(bucket, 0) + 1
    return totals


def bucket_table(cases: Sequence[ErrorCase], totals: dict[str, int]) -> str:
    """Report Table 6.4: how errors distribute across the structural buckets."""
    counts: dict[tuple[str, str], int] = {}
    for case in cases:
        for bucket in case.buckets:
            counts[(bucket, case.kind)] = counts.get((bucket, case.kind), 0) + 1

    buckets = sorted({bucket for bucket, _ in counts} | set(totals))
    lines = [
        "| Failure mode | Samples | False negatives | False positives | Error rate |",
        "|---|---|---|---|---|",
    ]
    for bucket in buckets:
        false_negatives = counts.get((bucket, "false_negative"), 0)
        false_positives = counts.get((bucket, "false_positive"), 0)
        total = totals.get(bucket, 0)
        rate = (false_negatives + false_positives) / total if total else 0.0
        lines.append(
            f"| {bucket} | {total} | {false_negatives} | {false_positives} | {rate:.1%} |"
        )
    return "\n".join(lines)


def render_report(
    cases: Sequence[ErrorCase],
    totals: dict[str, int],
    *,
    examples_per_bucket: int = 2,
) -> str:
    """Markdown for the error-analysis section, with worked examples per failure mode."""
    sections = [
        "# Error analysis",
        "",
        "Failure modes are structural properties of the sample — context length, numeric "
        "content, conjoined claims — assigned without reference to any judge's answer, so "
        "the categories were not drawn around the errors after the fact. A sample can belong "
        "to several, so the columns do not sum to the total.",
        "",
        "*Error rate* is per bucket: errors in that bucket over samples in that bucket. It "
        "is the column to read — a bucket with many errors simply because it is large tells "
        "you nothing.",
        "",
        bucket_table(cases, totals),
        "",
        "## Worked examples",
        "",
    ]

    by_bucket: dict[str, list[ErrorCase]] = {}
    for case in cases:
        for bucket in case.buckets:
            by_bucket.setdefault(bucket, []).append(case)

    for bucket in sorted(by_bucket):
        sections += [f"### {bucket}", ""]
        for case in by_bucket[bucket][:examples_per_bucket]:
            sections += [
                f"- `{case.dataset}/{case.sample_id}` ({case.series}) — "
                f"{case.kind.replace('_', ' ')}: human label "
                f"{'hallucinated' if case.label_hallucinated else 'faithful'}, "
                f"judge scored {case.faithfulness:.2f}",
            ]
        sections.append("")
    return "\n".join(sections)


@app.command()
def main(
    results: Annotated[list[Path], typer.Argument(help="Benchmark JSONL files.")],
    out: Annotated[Path, typer.Option(help="Directory for the error-analysis report.")] = Path(
        "report/errors"
    ),
    dataset: Annotated[
        str, typer.Option(help="Dataset the rows were scored from; joined for the source text.")
    ] = "ragtruth",
    data_dir: Annotated[Path | None, typer.Option(help="Dataset cache directory.")] = None,
    examples: Annotated[int, typer.Option(help="Worked examples per failure mode.")] = 2,
) -> None:
    """Classify every judge error by failure mode and write the report section."""
    frame = assign_series(load_rows(results))
    reports = {
        str(series): analyze_judge(frame[frame["series"] == series])
        for series in sorted(frame["series"].dropna().unique())
    }

    samples_by_id = {sample.id: sample for sample in load(dataset, data_dir=data_dir)}
    rows = _rows_with_buckets(frame, samples_by_id)
    if not rows:
        raise typer.BadParameter(
            f"No scored row matched a sample in {dataset!r}. Pass the --dataset the rows "
            "were produced from, and --data-dir if the cache is not in ./data."
        )
    cases = collect_errors(rows, reports)
    totals = bucket_totals(rows)

    out.mkdir(parents=True, exist_ok=True)
    report = render_report(cases, totals, examples_per_bucket=examples)
    (out / "error_analysis.md").write_text(report + "\n", encoding="utf-8")
    (out / "error_cases.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": case.sample_id,
                    "dataset": case.dataset,
                    "series": case.series,
                    "kind": case.kind,
                    "label_hallucinated": case.label_hallucinated,
                    "faithfulness": case.faithfulness,
                    "buckets": list(case.buckets),
                }
                for case in cases
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(report)
    typer.echo("")
    typer.echo(f"wrote {out / 'error_analysis.md'} ({len(cases)} error cases)")


def _rows_with_buckets(
    frame: Any,
    samples_by_id: dict[str, LabeledSample],
) -> list[dict[str, Any]]:
    """Attach a bucket set to each scored row by joining it back to its source sample.

    The join is against the dataset, not against anything in the rows file: the rows carry
    verdicts rather than source text, and estimating context length from the extracted
    claims would be inventing the very number the buckets are built on.
    """
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        sample = samples_by_id.get(str(record["sample_id"]))
        if sample is None:
            continue
        rows.append(
            {
                "sample_id": record["sample_id"],
                "dataset": record.get("dataset"),
                "series": record.get("series") or record.get("judge"),
                "faithfulness": record.get("faithfulness"),
                "label_hallucinated": record.get("label_hallucinated"),
                "buckets": sample_buckets(
                    contexts_chars=sum(len(text) for text in sample.contexts),
                    answer=sample.answer,
                    context_count=len(sample.contexts),
                ),
            }
        )
    return rows


if __name__ == "__main__":  # pragma: no cover
    app()
