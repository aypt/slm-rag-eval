"""Synthetic, span-labeled PII corpus for measuring the detector.

Every name, address, number and email here is invented. AGENTS.md forbids real personal
data in this repository, and a synthetic corpus is also the only practical way to get exact
span labels: hand-annotating real text would introduce annotation error into the very
measurement that is supposed to characterise the detector.

Documents are written as templates with `{slot}` markers, so the character offsets are
computed rather than counted by hand. A mistyped offset would silently become a false
negative and understate the detector — the failure mode this design removes.

Coverage is deliberately mixed:

* clean documents with no PII at all, which is the only way false positives can be measured;
* PII in the awkward positions detectors miss — sentence-initial, possessive, inside
  parentheses, adjacent to punctuation;
* near-miss text (product codes, dates, version numbers) that looks like PII but is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PIISample:
    """One document plus the exact character spans of the PII it contains."""

    id: str
    text: str
    spans: tuple[tuple[int, int, str], ...] = field(default_factory=tuple)
    note: str = ""


@dataclass(frozen=True)
class _Template:
    """A document with `{slot}` markers whose offsets are resolved at build time."""

    id: str
    template: str
    slots: tuple[tuple[str, str, str], ...]
    note: str = ""

    def build(self) -> PIISample:
        """Render the text and compute each slot's span, so offsets cannot drift."""
        text = self.template
        spans: list[tuple[int, int, str]] = []
        for name, value, entity_type in self.slots:
            marker = "{" + name + "}"
            index = text.index(marker)
            text = text[:index] + value + text[index + len(marker) :]
            spans.append((index, index + len(value), entity_type))
        return PIISample(id=self.id, text=text, spans=tuple(sorted(spans)), note=self.note)


_TEMPLATES: tuple[_Template, ...] = (
    _Template(
        id="clean-policy",
        template=(
            "The retention policy applies to all archived tickets. Records older than "
            "eighteen months are purged automatically at the end of each quarter."
        ),
        slots=(),
        note="No PII: any span flagged here is a false positive.",
    ),
    _Template(
        id="clean-technical",
        template=(
            "Deployment 4.2.1 failed twice on 2031-03-14 before the retry succeeded. "
            "Error code SRV-8842 indicates a transient upstream timeout, not data loss."
        ),
        slots=(),
        note="Version numbers, dates and codes that superficially resemble identifiers.",
    ),
    _Template(
        id="email-and-person",
        template="Please forward the audit summary to {person} at {email} before Friday.",
        slots=(
            ("person", "Marta Feldspar", "PERSON"),
            ("email", "marta.feldspar@example.com", "EMAIL_ADDRESS"),
        ),
    ),
    _Template(
        id="person-sentence-initial",
        template=(
            "{person} filed the complaint on behalf of the tenants' association. "
            "The association has not yet responded."
        ),
        slots=(("person", "Desmond Okonkwo", "PERSON"),),
        note="Sentence-initial names are a common miss for case-sensitive recognizers.",
    ),
    _Template(
        id="person-possessive",
        template="{person}'s account was suspended after the third failed verification.",
        slots=(("person", "Priya Raghunathan", "PERSON"),),
        note="Possessive form: the span must not swallow the apostrophe-s.",
    ),
    _Template(
        id="phone-parenthetical",
        template=(
            "The on-call engineer ({person}, reachable at {phone}) confirmed the outage "
            "window."
        ),
        slots=(
            ("person", "Yusuf Lindqvist", "PERSON"),
            ("phone", "(555) 010-4477", "PHONE_NUMBER"),
        ),
        note="PII inside parentheses, adjacent to punctuation on both sides.",
    ),
    _Template(
        id="location-and-person",
        template=(
            "{person} transferred from the {location} office last spring and now leads "
            "the migration team."
        ),
        slots=(
            ("person", "Anneliese Vroom", "PERSON"),
            ("location", "Thunder Bay", "LOCATION"),
        ),
    ),
    _Template(
        id="credit-card",
        template="The chargeback references card {card}, issued to {person}.",
        slots=(
            ("card", "4111 1111 1111 1111", "CREDIT_CARD"),
            ("person", "Halvard Ibsen", "PERSON"),
        ),
        note="Standard test card number: valid checksum, never a real account.",
    ),
    _Template(
        id="ip-address",
        template=(
            "Repeated authentication attempts originated from {ip} over a four-hour "
            "period."
        ),
        slots=(("ip", "203.0.113.42", "IP_ADDRESS"),),
        note="203.0.113.0/24 is the reserved documentation range (RFC 5737).",
    ),
    _Template(
        id="ssn",
        template="The benefits claim lists {ssn} as the taxpayer identifier for {person}.",
        slots=(
            ("ssn", "456-78-9012", "US_SSN"),
            ("person", "Corinne Baptiste", "PERSON"),
        ),
    ),
    _Template(
        id="multiple-people",
        template=(
            "{first} escalated the incident to {second}, who looped in {third} from the "
            "vendor side."
        ),
        slots=(
            ("first", "Tomasz Bielecki", "PERSON"),
            ("second", "Ngozi Adeyemi", "PERSON"),
            ("third", "Ruth Castellanos", "PERSON"),
        ),
        note="Three spans in one sentence: partial detection is visible as a partial score.",
    ),
    _Template(
        id="repeated-person",
        template=(
            "{first} approved the request. Later that week {second} reversed it without "
            "recording a reason."
        ),
        slots=(
            ("first", "Emeka Sandoval", "PERSON"),
            ("second", "Emeka Sandoval", "PERSON"),
        ),
        note="The same name twice: both occurrences must be masked, not just the first.",
    ),
    _Template(
        id="email-in-quotes",
        template='The form rejected "{email}" as malformed, though the address is valid.',
        slots=(("email", "n.oyelaran@example.org", "EMAIL_ADDRESS"),),
    ),
    _Template(
        id="mixed-clean-and-pii",
        template=(
            "Ticket SRV-2291 was reassigned three times before {person} resolved it. "
            "The root cause was a misconfigured retry budget, not a data issue."
        ),
        slots=(("person", "Beatriz Kowalczyk", "PERSON"),),
        note="One real span beside text that looks identifier-like but is not.",
    ),
    _Template(
        id="phone-plain",
        template="Callers were redirected to {phone} while the main line was down.",
        slots=(("phone", "555-010-9932", "PHONE_NUMBER"),),
    ),
    _Template(
        id="clean-numbers",
        template=(
            "Throughput rose from 1,200 to 4,850 requests per minute after the cache "
            "warm-up, and p95 latency fell to 180 ms."
        ),
        slots=(),
        note="Dense numerics with no PII: a detector that flags these loses precision.",
    ),
)


def labeled_corpus() -> list[PIISample]:
    """The full synthetic corpus, offsets computed from the templates."""
    return [template.build() for template in _TEMPLATES]


def corpus_span_count() -> int:
    """Total labeled PII spans, for the report's corpus description."""
    return sum(len(sample.spans) for sample in labeled_corpus())
