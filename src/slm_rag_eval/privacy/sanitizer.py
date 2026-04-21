"""Batch-stable PII sanitization built on Microsoft Presidio."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

DEFAULT_ENTITIES = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
    "US_SSN",
)


@dataclass(frozen=True)
class SanitizedBatch:
    """Sanitized texts and the in-memory-only values needed to restore display text."""

    texts: list[str]
    mapping: dict[str, str]


class _RecognizerResult(Protocol):
    entity_type: str
    start: int
    end: int
    score: float


class _Analyzer(Protocol):
    def analyze(
        self,
        *,
        text: str,
        language: str,
        entities: list[str],
        score_threshold: float,
    ) -> Sequence[_RecognizerResult]:
        """Return the detected entity spans above `score_threshold`."""
        ...


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    entity_type: str
    score: float


def _safe_entity_type(entity_type: str) -> str:
    normalized = re.sub(r"[^A-Z0-9_]+", "_", entity_type.upper()).strip("_")
    return normalized or "PII"


def _merge_overlapping_spans(spans: list[_Span]) -> list[_Span]:
    """Merge overlaps so a lower-priority overlap cannot leave part of PII exposed."""
    if not spans:
        return []

    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[_Span] = []
    current = ordered[0]
    for candidate in ordered[1:]:
        if candidate.start >= current.end:
            merged.append(current)
            current = candidate
            continue

        preferred = max((current, candidate), key=lambda span: span.score)
        current = _Span(
            start=current.start,
            end=max(current.end, candidate.end),
            entity_type=preferred.entity_type,
            score=preferred.score,
        )
    merged.append(current)
    return merged


def restore(text: str, mapping: Mapping[str, str]) -> str:
    """Restore placeholders in judge-produced text for trusted local display."""
    if not mapping:
        return text

    placeholder_pattern = re.compile(
        "|".join(re.escape(placeholder) for placeholder in sorted(mapping, key=len, reverse=True))
    )
    return placeholder_pattern.sub(lambda match: mapping[match.group(0)], text)


class Sanitizer:
    """Detects PII and replaces it with stable placeholders before any judge call."""

    def __init__(
        self,
        *,
        entities: Sequence[str] | None = None,
        score_threshold: float = 0.4,
        analyzer: _Analyzer | None = None,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        self.entities = list(entities if entities is not None else DEFAULT_ENTITIES)
        self.score_threshold = score_threshold
        self._analyzer = analyzer

    def _get_analyzer(self) -> _Analyzer:
        if self._analyzer is not None:
            return self._analyzer

        try:
            from presidio_analyzer import AnalyzerEngine

            self._analyzer = AnalyzerEngine()
        except (ImportError, OSError, ValueError) as exc:
            raise RuntimeError(
                "Presidio and its en_core_web_lg spaCy model are required for privacy masking; "
                "run `make setup` before evaluating with privacy_mode='mask'"
            ) from exc
        return self._analyzer

    def _detect(self, analyzer: _Analyzer, text: str) -> list[_Span]:
        if not text:
            return []
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=self.entities,
            score_threshold=self.score_threshold,
        )
        spans = [
            _Span(
                start=result.start,
                end=result.end,
                entity_type=_safe_entity_type(result.entity_type),
                score=result.score,
            )
            for result in results
            if 0 <= result.start < result.end <= len(text)
        ]
        return _merge_overlapping_spans(spans)

    def sanitize(self, texts: list[str]) -> SanitizedBatch:
        """Mask configured PII consistently across all texts in a single batch."""
        if not texts:
            return SanitizedBatch(texts=[], mapping={})

        analyzer = self._get_analyzer()
        detected = [self._detect(analyzer, text) for text in texts]
        occupied_text = "\n".join(texts)
        counters: defaultdict[str, int] = defaultdict(int)
        original_to_placeholder: dict[str, str] = {}
        mapping: dict[str, str] = {}

        for text, spans in zip(texts, detected, strict=True):
            for span in spans:
                original = text[span.start : span.end]
                if original in original_to_placeholder:
                    continue

                entity_type = span.entity_type
                while True:
                    counters[entity_type] += 1
                    placeholder = f"<{entity_type}_{counters[entity_type]}>"
                    if placeholder not in occupied_text and placeholder not in mapping:
                        break
                original_to_placeholder[original] = placeholder
                mapping[placeholder] = original

        sanitized_texts: list[str] = []
        for text, spans in zip(texts, detected, strict=True):
            sanitized = text
            for span in reversed(spans):
                original = text[span.start : span.end]
                sanitized = (
                    sanitized[: span.start]
                    + original_to_placeholder[original]
                    + sanitized[span.end :]
                )
            sanitized_texts.append(sanitized)

        return SanitizedBatch(texts=sanitized_texts, mapping=mapping)

    def restore(self, text: str, mapping: Mapping[str, str]) -> str:
        """Restore placeholders in judge-produced text for trusted local display."""
        return restore(text, mapping)
