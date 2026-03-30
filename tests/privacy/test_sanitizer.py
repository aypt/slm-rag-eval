from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pytest

from slm_rag_eval.privacy.sanitizer import Sanitizer, restore


@dataclass(frozen=True)
class StubResult:
    entity_type: str
    start: int
    end: int
    score: float = 0.9


class StubAnalyzer:
    def __init__(self, detections: dict[str, str]) -> None:
        self.detections = detections
        self.calls: list[dict[str, object]] = []

    def analyze(
        self,
        *,
        text: str,
        language: str,
        entities: list[str],
        score_threshold: float,
    ) -> list[StubResult]:
        self.calls.append(
            {
                "text": text,
                "language": language,
                "entities": entities,
                "score_threshold": score_threshold,
            }
        )
        results: list[StubResult] = []
        for surface, entity_type in self.detections.items():
            start = text.find(surface)
            while start >= 0:
                results.append(
                    StubResult(
                        entity_type=entity_type,
                        start=start,
                        end=start + len(surface),
                    )
                )
                start = text.find(surface, start + len(surface))
        return results


def test_sanitize_uses_stable_batch_placeholders_and_forwards_configuration() -> None:
    analyzer = StubAnalyzer(
        {
            "Morgan Fable": "PERSON",
            "morgan@example.com": "EMAIL_ADDRESS",
        }
    )
    sanitizer = Sanitizer(
        entities=["PERSON", "EMAIL_ADDRESS"],
        score_threshold=0.55,
        analyzer=analyzer,
    )

    batch = sanitizer.sanitize(
        [
            "Ask Morgan Fable for an update.",
            "Morgan Fable uses morgan@example.com.",
        ]
    )

    assert batch.texts == [
        "Ask <PERSON_1> for an update.",
        "<PERSON_1> uses <EMAIL_ADDRESS_1>.",
    ]
    assert batch.mapping == {
        "<PERSON_1>": "Morgan Fable",
        "<EMAIL_ADDRESS_1>": "morgan@example.com",
    }
    assert all(call["language"] == "en" for call in analyzer.calls)
    assert all(call["entities"] == ["PERSON", "EMAIL_ADDRESS"] for call in analyzer.calls)
    assert all(call["score_threshold"] == 0.55 for call in analyzer.calls)


def test_sanitize_avoids_colliding_with_existing_placeholder_text() -> None:
    analyzer = StubAnalyzer({"River Quill": "PERSON"})
    sanitizer = Sanitizer(analyzer=analyzer)

    batch = sanitizer.sanitize(["<PERSON_1> is a label; River Quill is synthetic."])

    assert batch.texts == ["<PERSON_1> is a label; <PERSON_2> is synthetic."]
    assert batch.mapping == {"<PERSON_2>": "River Quill"}


def test_restore_is_exact_and_does_not_cascade_replacements() -> None:
    mapping = {
        "<PERSON_1>": "<EMAIL_ADDRESS_1>",
        "<EMAIL_ADDRESS_1>": "synthetic@example.com",
    }

    assert restore("Contact <PERSON_1>.", mapping) == "Contact <EMAIL_ADDRESS_1>."


_PRESIDIO_READY = all(
    importlib.util.find_spec(package) is not None
    for package in ("presidio_analyzer", "presidio_anonymizer", "spacy", "en_core_web_lg")
)


@pytest.mark.skipif(
    not _PRESIDIO_READY,
    reason="Presidio integration requires its packages and the en_core_web_lg spaCy model",
)
def test_presidio_detects_synthetic_email_and_phone() -> None:
    sanitizer = Sanitizer(entities=["EMAIL_ADDRESS", "PHONE_NUMBER"])
    email = "privacy.user@example.com"
    phone = "202-555-0147"

    batch = sanitizer.sanitize([f"Write to {email} or call {phone}."])

    assert email not in batch.texts[0]
    assert phone not in batch.texts[0]
    assert set(batch.mapping.values()) == {email, phone}
