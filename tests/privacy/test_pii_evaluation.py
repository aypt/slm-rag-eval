"""The PII measurement is itself a measurement, so its arithmetic is pinned here.

Recall is the number the privacy claim in the report rests on. If this harness overstates
it, the report makes a safety claim the system does not support — the most consequential
way a number here could be wrong.
"""

from __future__ import annotations

import pytest

from slm_rag_eval.privacy.evaluation import (
    DetectionMetrics,
    Span,
    detection_misses,
    evaluate_detection,
    match_spans,
    render_report,
)
from slm_rag_eval.privacy.fixtures import PIISample, corpus_span_count, labeled_corpus


def test_every_corpus_span_lands_on_real_text() -> None:
    """Offsets are computed from templates; a drifted one would look like a detector miss."""
    for sample in labeled_corpus():
        for start, end, entity_type in sample.spans:
            snippet = sample.text[start:end]
            assert snippet, f"{sample.id}: empty span for {entity_type}"
            assert snippet == snippet.strip(), f"{sample.id}: {snippet!r} has loose whitespace"


def test_the_corpus_contains_clean_documents_so_precision_is_measurable() -> None:
    """With no PII-free text, a detector that flags everything would score perfectly."""
    clean = [sample for sample in labeled_corpus() if not sample.spans]

    assert len(clean) >= 3
    assert corpus_span_count() >= 15


def test_an_overlapping_detection_counts_as_caught() -> None:
    """"Dr. Alvarez" over a label of "Alvarez" protected the data; it is not a miss."""
    result = match_spans([Span(4, 11, "PERSON")], [Span(0, 11, "PERSON")])

    assert len(result.matched) == 1
    assert result.missed == ()


def test_the_wrong_entity_type_still_counts_as_masked_by_default() -> None:
    """Masked as PHONE_NUMBER instead of US_SSN is still masked."""
    untyped = match_spans([Span(0, 5, "US_SSN")], [Span(0, 5, "PHONE_NUMBER")])
    typed = match_spans([Span(0, 5, "US_SSN")], [Span(0, 5, "PHONE_NUMBER")], typed=True)

    assert len(untyped.matched) == 1
    assert typed.missed == (Span(0, 5, "US_SSN"),)


def test_a_missed_span_is_reported_as_missed_not_as_a_false_alarm() -> None:
    result = match_spans([Span(0, 5, "PERSON")], [Span(20, 25, "PERSON")])

    assert result.missed == (Span(0, 5, "PERSON"),)
    assert result.spurious == (Span(20, 25, "PERSON"),)


def test_per_type_counts_sum_to_the_overall_row() -> None:
    """Two passes with different rules produced tables that did not reconcile."""
    samples = [
        PIISample(id="a", text="x" * 40, spans=((0, 5, "PERSON"), (10, 15, "EMAIL_ADDRESS"))),
        PIISample(id="b", text="y" * 40, spans=((0, 5, "PERSON"),)),
    ]

    def detect(text: str) -> list[Span]:
        return [Span(0, 5, "PERSON"), Span(30, 35, "LOCATION")]

    results = evaluate_detection(samples, detect)

    overall = results["ALL"]
    per_type = [value for key, value in results.items() if key != "ALL"]
    assert overall.true_positives == sum(m.true_positives for m in per_type)
    assert overall.false_positives == sum(m.false_positives for m in per_type)
    assert overall.false_negatives == sum(m.false_negatives for m in per_type)


def test_recall_and_precision_are_computed_the_standard_way() -> None:
    metrics = DetectionMetrics("PERSON", true_positives=3, false_positives=1, false_negatives=1)

    assert metrics.precision == pytest.approx(0.75)
    assert metrics.recall == pytest.approx(0.75)
    assert metrics.support == 4


def test_a_detector_that_finds_nothing_scores_zero_recall_not_one() -> None:
    samples = [PIISample(id="a", text="x" * 20, spans=((0, 5, "PERSON"),))]

    results = evaluate_detection(samples, lambda text: [])

    assert results["ALL"].recall == pytest.approx(0.0)


def test_missed_spans_are_listed_for_the_error_analysis() -> None:
    samples = [PIISample(id="doc", text="z" * 20, spans=((0, 5, "LOCATION"),))]

    misses = detection_misses(samples, lambda text: [])

    assert [miss["entity_type"] for miss in misses] == ["LOCATION"]
    assert misses[0]["sample_id"] == "doc"


def test_the_rendered_report_names_recall_as_the_privacy_number() -> None:
    results = evaluate_detection(
        [PIISample(id="a", text="x" * 20, spans=((0, 5, "PERSON"),))],
        lambda text: [Span(0, 5, "PERSON")],
    )

    report = render_report(results, corpus_size=1)

    assert "Recall is the privacy-relevant number" in report
    assert "| PERSON |" in report


def test_the_corpus_meets_the_plan_that_was_registered_before_it_was_written() -> None:
    """The plan is a floor set in advance; the corpus is never trimmed to flatter a score.

    This test is what keeps that promise checkable. If a future change drops a category
    below its floor, that is visible here rather than in a quietly weaker recall number.
    """
    from collections import Counter

    from slm_rag_eval.privacy.fixtures import CORPUS_PLAN

    corpus = labeled_corpus()
    counts = Counter(entity for sample in corpus for _, _, entity in sample.spans)
    clean = sum(1 for sample in corpus if not sample.spans)

    assert len(corpus) >= CORPUS_PLAN["documents"]
    assert clean >= CORPUS_PLAN["clean_documents"]
    for entity, floor in CORPUS_PLAN.items():
        if entity in {"documents", "clean_documents"}:
            continue
        assert counts[entity] >= floor, f"{entity}: {counts[entity]} < planned floor {floor}"


def test_every_document_has_a_unique_id() -> None:
    ids = [sample.id for sample in labeled_corpus()]

    assert len(ids) == len(set(ids))


def test_the_corpus_covers_the_positions_detectors_are_known_to_miss() -> None:
    """Sentence-initial, possessive and parenthetical placement, plus repeated entities."""
    ids = {sample.id for sample in labeled_corpus()}

    for required in (
        "person-sentence-initial",
        "person-possessive",
        "phone-parenthetical",
        "repeated-person",
        "ip-ipv6",
        "email-plus-tag",
    ):
        assert required in ids
