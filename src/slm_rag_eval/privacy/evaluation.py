"""Measure the privacy layer: does it catch PII, and what does catching it cost?

Two questions the report has to answer with numbers, and they pull in opposite directions:

* **Detection.** Over a labeled corpus, which PII spans does the detector find? Reported as
  precision, recall and F1 at the span level, per entity type. Recall is the number that
  matters for a privacy claim — a missed span is PII that left the machine — so it is
  reported separately from precision rather than hidden inside an F1.
* **Cost.** Masking replaces names with placeholders, and a judge reasoning over
  `<PERSON_1>` may score differently from one reading "Dr. Alvarez". That is measured by
  the mask-on/mask-off ablation in `bench.analyze`, not here.

The corpus is synthetic (`privacy/fixtures.py`). Real PII must never enter this repository,
and a synthetic corpus is also the only way to have exact span labels to score against.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.privacy.fixtures import PIISample, labeled_corpus

app = typer.Typer(help=__doc__, add_completion=False)


@dataclass(frozen=True)
class Span:
    """One PII occurrence: where it is and what kind it is."""

    start: int
    end: int
    entity_type: str

    def overlaps(self, other: Span) -> bool:
        """Whether two spans cover any character in common."""
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class DetectionMetrics:
    """Span-level detection quality for one entity type, or for everything at once."""

    entity_type: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Of the spans flagged, how many were really PII. 1.0 when nothing was flagged."""
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 1.0

    @property
    def recall(self) -> float:
        """Of the real PII spans, how many were caught. The number a privacy claim rests on."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 1.0

    @property
    def f1(self) -> float:
        """Harmonic mean; 0.0 when precision and recall are both zero."""
        total = self.precision + self.recall
        return 0.0 if total == 0 else 2 * self.precision * self.recall / total

    @property
    def support(self) -> int:
        """How many true spans of this type the corpus contains."""
        return self.true_positives + self.false_negatives


@dataclass(frozen=True)
class MatchResult:
    """One document's outcome: which expected spans were caught and which were not."""

    matched: tuple[tuple[Span, Span], ...]
    missed: tuple[Span, ...]
    spurious: tuple[Span, ...]


def match_spans(
    expected: Sequence[Span],
    detected: Sequence[Span],
    *,
    typed: bool = False,
) -> MatchResult:
    """Greedily pair detected spans with expected ones.

    Overlap rather than exact boundaries is the right criterion for a privacy metric: a
    detector that flags "Dr. Alvarez" where the label says "Alvarez" has protected the data,
    and scoring that as both a miss and a false alarm would understate the layer twice over.

    With `typed=False` a span counts as caught whatever label the detector gave it, because
    masked-as-the-wrong-type is still masked. `typed=True` additionally requires the label
    to match, which measures classification rather than protection.
    """
    unmatched = list(expected)
    matched: list[tuple[Span, Span]] = []
    spurious: list[Span] = []
    for candidate in detected:
        for index, target in enumerate(unmatched):
            if candidate.overlaps(target) and (
                not typed or candidate.entity_type == target.entity_type
            ):
                matched.append((target, candidate))
                unmatched.pop(index)
                break
        else:
            spurious.append(candidate)
    return MatchResult(tuple(matched), tuple(unmatched), tuple(spurious))


def evaluate_detection(
    samples: Sequence[PIISample],
    detect: Any,
    *,
    typed: bool = False,
) -> dict[str, DetectionMetrics]:
    """Score a detector over the corpus, overall and per entity type.

    Every row comes from one matching pass, attributed by span: a caught or missed span
    counts under the type the *label* gives it, and a spurious detection under the type the
    *detector* gave it. The per-type rows therefore add up to the ALL row exactly — two
    passes with different rules produced tables that did not reconcile, which is worse than
    useless in a report a reader is meant to check.
    """
    per_type: dict[str, list[int]] = {}
    overall = [0, 0, 0]

    def bucket(entity_type: str) -> list[int]:
        return per_type.setdefault(entity_type, [0, 0, 0])

    for sample in samples:
        expected = [Span(*span) for span in sample.spans]
        result = match_spans(expected, list(detect(sample.text)), typed=typed)

        for target, _ in result.matched:
            bucket(target.entity_type)[0] += 1
            overall[0] += 1
        for candidate in result.spurious:
            bucket(candidate.entity_type)[1] += 1
            overall[1] += 1
        for target in result.missed:
            bucket(target.entity_type)[2] += 1
            overall[2] += 1

    results = {
        entity_type: DetectionMetrics(entity_type, *counts)
        for entity_type, counts in sorted(per_type.items())
    }
    results["ALL"] = DetectionMetrics("ALL", *overall)
    return results


def detection_misses(samples: Sequence[PIISample], detect: Any) -> list[dict[str, str]]:
    """Every span the detector missed, with its document — the report's error analysis.

    A recall figure says how much leaked; this says what kind of thing leaks, which is the
    part a reader can act on.
    """
    misses: list[dict[str, str]] = []
    for sample in samples:
        expected = [Span(*span) for span in sample.spans]
        result = match_spans(expected, list(detect(sample.text)))
        for span in result.missed:
            misses.append(
                {
                    "sample_id": sample.id,
                    "entity_type": span.entity_type,
                    "text": sample.text[span.start : span.end],
                    "context": sample.text,
                }
            )
    return misses


def presidio_detector(settings: Settings) -> Any:
    """A detect(text) -> [Span] backed by the same Sanitizer the pipeline uses.

    Deliberately the production object with the production settings: measuring a detector
    configured differently from the one that runs would report a number the deployment
    never achieves.
    """
    from slm_rag_eval.metrics.registry import build_sanitizer

    sanitizer = build_sanitizer(settings)

    def detect(text: str) -> list[Span]:
        analyzer = sanitizer._get_analyzer()  # noqa: SLF001 -- measuring the real detector
        return [
            Span(span.start, span.end, span.entity_type)
            for span in sanitizer._detect(analyzer, text)  # noqa: SLF001
        ]

    return detect


def render_report(
    results: dict[str, DetectionMetrics],
    *,
    corpus_size: int,
    misses: Sequence[dict[str, str]] = (),
) -> str:
    """Report Table 6.3 (detection half): per-entity precision, recall and F1."""
    lines = [
        "### PII detection quality",
        "",
        f"Synthetic corpus: {corpus_size} documents. Span-level, overlap-based matching.",
        "",
        "| Entity type | Support | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(results, key=lambda key: (key == "ALL", key)):
        metrics = results[name]
        lines.append(
            f"| {name} | {metrics.support} | {metrics.true_positives} | "
            f"{metrics.false_positives} | {metrics.false_negatives} | "
            f"{metrics.precision:.3f} | {metrics.recall:.3f} | {metrics.f1:.3f} |"
        )
    lines += [
        "",
        "Recall is the privacy-relevant number: every false negative is PII that would have "
        "reached the judge. Precision costs utility rather than privacy — an over-eager "
        "detector masks text the judge then cannot reason about, which is what the "
        "mask-on/mask-off ablation measures.",
        "",
        "A span counts as caught when a detection overlaps it, whatever type the detector "
        "assigned: masked as the wrong type is still masked. Per-type rows sum to the ALL "
        "row, with caught and missed spans attributed to the labeled type and spurious "
        "detections to the detected type.",
    ]
    if misses:
        lines += [
            "",
            "#### Missed spans",
            "",
            "| Entity type | Text | Document |",
            "|---|---|---|",
        ]
        lines += [
            f"| {miss['entity_type']} | `{miss['text']}` | {miss['sample_id']} |"
            for miss in misses
        ]
    return "\n".join(lines)


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Directory for the report and results.")] = Path(
        "report/privacy"
    ),
    typed: Annotated[
        bool, typer.Option("--typed", help="Also require the entity type to match.")
    ] = False,
) -> None:
    """Score the configured PII detector over the synthetic corpus."""
    settings = get_settings()
    samples = labeled_corpus()
    detector = presidio_detector(settings)
    results = evaluate_detection(samples, detector, typed=typed)

    out.mkdir(parents=True, exist_ok=True)
    misses = detection_misses(samples, detector)
    report = render_report(results, corpus_size=len(samples), misses=misses)
    (out / "pii_detection.md").write_text(report + "\n", encoding="utf-8")
    (out / "pii_detection.json").write_text(
        json.dumps(
            {
                "corpus_size": len(samples),
                "typed_matching": typed,
                "entities": settings.privacy_entities,
                "score_threshold": settings.privacy_score_threshold,
                "results": {
                    name: {
                        "support": metrics.support,
                        "true_positives": metrics.true_positives,
                        "false_positives": metrics.false_positives,
                        "false_negatives": metrics.false_negatives,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                    }
                    for name, metrics in results.items()
                },
                "misses": [
                    {k: v for k, v in miss.items() if k != "context"} for miss in misses
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(report)
    typer.echo("")
    typer.echo(f"wrote {out / 'pii_detection.md'} and {out / 'pii_detection.json'}")


if __name__ == "__main__":  # pragma: no cover
    app()
