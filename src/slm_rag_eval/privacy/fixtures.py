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

**The plan in `CORPUS_PLAN` was fixed before the corpus was written, and the corpus is not
edited in response to what the detector scores.** That rule is the whole reason this number
can go in a report: a corpus adjusted until the detector looks good measures the author, not
the detector. A miss is a result to report, not a case to remove — the one location the
detector misses is in the report for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CORPUS_PLAN: dict[str, int] = {
    "documents": 60,
    "clean_documents": 14,
    "PERSON": 34,
    "EMAIL_ADDRESS": 11,
    "PHONE_NUMBER": 10,
    "LOCATION": 12,
    "CREDIT_CARD": 6,
    "IP_ADDRESS": 7,
    "US_SSN": 6,
}
"""Minimum composition, fixed before the corpus was written. Tested, never retuned.

These are floors, not exact counts: documents were written to satisfy the plan, and a
category may end up above its floor because one document carries several entity types.

Roughly a quarter of the documents carry no PII: precision is only measurable against text
the detector should leave alone, and a corpus of nothing but positives cannot distinguish a
careful detector from one that flags every capitalised word.
"""


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
    # --- clean negatives -------------------------------------------------------------
    _Template(
        id="clean-meeting",
        template=(
            "The steering group agreed to defer the migration until the audit closes. "
            "Minutes will circulate to the distribution list on Monday."
        ),
        slots=(),
        note="Ordinary corporate prose with capitalised words that are not names.",
    ),
    _Template(
        id="clean-capitalised-nouns",
        template=(
            "The Finance Committee approved the Data Retention Standard, and the Security "
            "Working Group will publish implementation guidance next quarter."
        ),
        slots=(),
        note="Title-case entity names: a bare capitalisation heuristic flags all of these.",
    ),
    _Template(
        id="clean-product-codes",
        template=(
            "Batch AX-4417 shipped alongside batch AX-4418. The packing note references "
            "purchase order PO-99120 and container MSKU-7741208."
        ),
        slots=(),
        note="Identifier-shaped codes that are not personal identifiers.",
    ),
    _Template(
        id="clean-monetary",
        template=(
            "The variance was 4,200 dollars against a budget of 118,000 dollars, driven "
            "almost entirely by egress charges in the final week."
        ),
        slots=(),
        note="Grouped digits that resemble card or account numbers.",
    ),
    _Template(
        id="clean-dates",
        template=(
            "The window opens 03/04/2031 and closes 05/06/2031. A freeze applies between "
            "12-24-2030 and 01-02-2031 inclusive."
        ),
        slots=(),
        note="Date formats that overlap SSN and phone punctuation patterns.",
    ),
    _Template(
        id="clean-versions",
        template=(
            "Rollback from 12.4.1 to 12.3.9 resolved the regression. The affected build "
            "was 192.168 megabytes, larger than the previous release."
        ),
        slots=(),
        note="A dotted quad in prose that is a size, not an IP address.",
    ),
    _Template(
        id="clean-place-adjectives",
        template=(
            "The team adopted a Canadian spelling convention and a European date format "
            "for all customer-facing templates."
        ),
        slots=(),
        note="Demonyms rather than locations; masking them would damage meaning for nothing.",
    ),
    _Template(
        id="clean-technical-hosts",
        template=(
            "Traffic moved from the staging cluster to the production cluster without "
            "downtime. Both run behind the same load balancer pool."
        ),
        slots=(),
        note="Infrastructure nouns with no identifiers at all.",
    ),
    _Template(
        id="clean-quantities",
        template=(
            "Retention rose to 91 percent after the onboarding change, from 78 percent in "
            "the preceding eight-week period."
        ),
        slots=(),
        note="Percentages and durations.",
    ),
    _Template(
        id="clean-policy-two",
        template=(
            "Access reviews run twice a year. Reviewers must record a justification for "
            "every retained permission, and unjustified permissions expire automatically."
        ),
        slots=(),
        note="Policy prose, no entities.",
    ),
    _Template(
        id="clean-abbreviations",
        template=(
            "The SLA covers API availability but excludes scheduled maintenance. The RTO "
            "is four hours and the RPO is fifteen minutes."
        ),
        slots=(),
        note="Dense acronyms, which some recognizers mistake for initials.",
    ),
    _Template(
        id="clean-ticket-thread",
        template=(
            "Ticket INC-88213 was merged into INC-88190 after triage confirmed a shared "
            "root cause in the retry scheduler."
        ),
        slots=(),
        note="Ticket identifiers only.",
    ),
    # --- people in awkward positions -------------------------------------------------
    _Template(
        id="person-hyphenated",
        template="{person} chaired the review and signed the closing memorandum.",
        slots=(("person", "Ana-Lucia Petrescu", "PERSON"),),
        note="Hyphenated given name.",
    ),
    _Template(
        id="person-with-title",
        template="The report was countersigned by Dr. {person} before submission.",
        slots=(("person", "Ifeoma Achebe", "PERSON"),),
        note="Honorific prefix: the span may or may not include the title.",
    ),
    _Template(
        id="person-initials",
        template="{person} approved the exception on the second attempt.",
        slots=(("person", "R. J. Mkhize", "PERSON"),),
        note="Initials with periods, which tokenizers split awkwardly.",
    ),
    _Template(
        id="person-list",
        template="Attendees were {first}, {second}, and {third}.",
        slots=(
            ("first", "Soren Vasquez", "PERSON"),
            ("second", "Mei-Ling Toft", "PERSON"),
            ("third", "Aleksander Nowak", "PERSON"),
        ),
        note="Comma-separated list, where boundary errors merge adjacent names.",
    ),
    _Template(
        id="person-end-of-sentence",
        template="The escalation was finally resolved by {person}.",
        slots=(("person", "Georgios Pappas", "PERSON"),),
        note="Name immediately before a full stop.",
    ),
    _Template(
        id="person-quoted-speech",
        template='"The rollback is complete," wrote {person} in the incident channel.',
        slots=(("person", "Fatima Zahra Idrissi", "PERSON"),),
        note="Three-token name inside quoted speech.",
    ),
    _Template(
        id="person-apostrophe-surname",
        template="{person} filed the variance request on behalf of the night shift.",
        slots=(("person", "Siobhan O'Callaghan", "PERSON"),),
        note="Apostrophe inside the surname itself.",
    ),
    _Template(
        id="person-and-location",
        template="{person} relocated to {location} to open the second support desk.",
        slots=(
            ("person", "Lars Bergqvist", "PERSON"),
            ("location", "Reykjavik", "LOCATION"),
        ),
    ),
    _Template(
        id="person-parenthetical-role",
        template="The change was authorised by {person} (platform reliability) last Thursday.",
        slots=(("person", "Chidinma Eze", "PERSON"),),
    ),
    _Template(
        id="person-possessive-plural",
        template="{person}' notes from the retrospective were circulated unedited.",
        slots=(("person", "Nikos Stamatis", "PERSON"),),
        note="Trailing apostrophe with no s.",
    ),
    # --- contact details -------------------------------------------------------------
    _Template(
        id="email-plus-tag",
        template="Escalations route to {email} outside business hours.",
        slots=(("email", "oncall+platform@example.com", "EMAIL_ADDRESS"),),
        note="Plus-addressing, which some regexes reject.",
    ),
    _Template(
        id="email-subdomain",
        template="The bounce originated from {email} and was retried twice.",
        slots=(("email", "n.okafor@mail.corp.example.net", "EMAIL_ADDRESS"),),
        note="Multi-label domain.",
    ),
    _Template(
        id="email-and-phone",
        template="Reach the duty manager at {email} or {phone}.",
        slots=(
            ("email", "duty.manager@example.org", "EMAIL_ADDRESS"),
            ("phone", "+1 555 010 8842", "PHONE_NUMBER"),
        ),
        note="International dialling format.",
    ),
    _Template(
        id="email-trailing-punctuation",
        template="Send the signed copy to {email}, then archive the thread.",
        slots=(("email", "records@example.com", "EMAIL_ADDRESS"),),
        note="Comma immediately after the address.",
    ),
    _Template(
        id="email-uppercase",
        template="The form accepted {email} despite the unusual capitalisation.",
        slots=(("email", "T.Bergstrom@Example.COM", "EMAIL_ADDRESS"),),
        note="Mixed case, which a case-sensitive pattern misses.",
    ),
    _Template(
        id="phone-extension",
        template="Dial {phone} and ask for the escalation desk.",
        slots=(("phone", "555-010-3391", "PHONE_NUMBER"),),
    ),
    _Template(
        id="phone-dotted",
        template="The vendor listed {phone} on the invoice footer.",
        slots=(("phone", "555.010.7723", "PHONE_NUMBER"),),
        note="Dot-separated, which overlaps version and IP patterns.",
    ),
    _Template(
        id="phone-and-person",
        template="{person} can be reached on {phone} until 18:00.",
        slots=(
            ("person", "Hiroshi Nakamura", "PERSON"),
            ("phone", "(555) 010-2214", "PHONE_NUMBER"),
        ),
    ),
    _Template(
        id="phone-two-numbers",
        template="Primary {first} failed over to secondary {second} during the outage.",
        slots=(
            ("first", "555-010-4400", "PHONE_NUMBER"),
            ("second", "555-010-4401", "PHONE_NUMBER"),
        ),
        note="Adjacent near-identical numbers, where a greedy span merges both.",
    ),
    # --- structured identifiers ------------------------------------------------------
    _Template(
        id="card-hyphenated",
        template="The refund was issued to card {card} on the same day.",
        slots=(("card", "5555-5555-5555-4444", "CREDIT_CARD"),),
        note="Standard test card, hyphen separated.",
    ),
    _Template(
        id="card-unspaced",
        template="Reconciliation matched the settlement to card {card}.",
        slots=(("card", "4012888888881881", "CREDIT_CARD"),),
        note="Standard test card with no separators.",
    ),
    _Template(
        id="card-and-person",
        template="{person} disputed the charge on card {card}.",
        slots=(
            ("person", "Bethany Oduya", "PERSON"),
            ("card", "4111111111111111", "CREDIT_CARD"),
        ),
    ),
    _Template(
        id="card-partial-context",
        template="The statement shows card {card} and a matching authorisation code.",
        slots=(("card", "5105 1051 0510 5100", "CREDIT_CARD"),),
        note="Standard test card, space separated.",
    ),
    _Template(
        id="ip-two-addresses",
        template="Traffic from {first} was proxied through {second} before reaching the API.",
        slots=(
            ("first", "198.51.100.17", "IP_ADDRESS"),
            ("second", "203.0.113.9", "IP_ADDRESS"),
        ),
        note="Both from reserved documentation ranges (RFC 5737).",
    ),
    _Template(
        id="ip-with-port",
        template="The listener bound to {ip} and refused connections from elsewhere.",
        slots=(("ip", "192.0.2.44", "IP_ADDRESS"),),
        note="TEST-NET-1 documentation range.",
    ),
    _Template(
        id="ip-ipv6",
        template="The dual-stack host answered on {ip} during the migration window.",
        slots=(("ip", "2001:db8::8a2e:370:7334", "IP_ADDRESS"),),
        note="IPv6 documentation prefix (RFC 3849); often missed by IPv4-only patterns.",
    ),
    _Template(
        id="ip-and-person",
        template="{person} traced the anomaly to {ip} within the hour.",
        slots=(
            ("person", "Tobias Lindgren", "PERSON"),
            ("ip", "198.51.100.203", "IP_ADDRESS"),
        ),
    ),
    _Template(
        id="ssn-spaces",
        template="The tax record lists {ssn} for the beneficiary.",
        slots=(("ssn", "123 45 6789", "US_SSN"),),
        note="Space-separated SSN.",
    ),
    _Template(
        id="ssn-unformatted",
        template="The intake form recorded {ssn} without separators.",
        slots=(("ssn", "234567890", "US_SSN"),),
        note="Nine bare digits, ambiguous with other identifiers.",
    ),
    _Template(
        id="ssn-and-person",
        template="{person} corrected the identifier from the earlier filing to {ssn}.",
        slots=(
            ("person", "Delphine Moreau", "PERSON"),
            ("ssn", "345-67-8901", "US_SSN"),
        ),
    ),
    _Template(
        id="ssn-in-list",
        template="Fields captured: name {person}, identifier {ssn}, region {location}.",
        slots=(
            ("person", "Kwame Asante", "PERSON"),
            ("ssn", "567-89-0123", "US_SSN"),
            ("location", "Manitoba", "LOCATION"),
        ),
        note="Three entity types in one dense line.",
    ),
    # --- locations -------------------------------------------------------------------
    _Template(
        id="location-city-country",
        template="The disaster-recovery site is in {location}, and failover is automatic.",
        slots=(("location", "Trondheim", "LOCATION"),),
    ),
    _Template(
        id="location-two-cities",
        template="Latency between {first} and {second} exceeded the agreed threshold.",
        slots=(
            ("first", "Auckland", "LOCATION"),
            ("second", "Osaka", "LOCATION"),
        ),
    ),
    _Template(
        id="location-street",
        template="Deliveries go to the {location} depot until the lease ends.",
        slots=(("location", "Rotterdam", "LOCATION"),),
    ),
    _Template(
        id="location-region",
        template="The rollout covered {location} before any other region.",
        slots=(("location", "Saskatchewan", "LOCATION"),),
        note="Sub-national region name.",
    ),
    _Template(
        id="location-and-email",
        template="The {location} office publishes its rota to {email} every Friday.",
        slots=(
            ("location", "Ljubljana", "LOCATION"),
            ("email", "rota@example.org", "EMAIL_ADDRESS"),
        ),
    ),
    _Template(
        id="location-uncommon",
        template="Field engineers were dispatched from {location} after the storm.",
        slots=(("location", "Yellowknife", "LOCATION"),),
        note="Less common place name; the kind NER is likelier to miss than a capital city.",
    ),
    # --- mixed density ---------------------------------------------------------------
    _Template(
        id="dense-record",
        template=(
            "Record: {person}, {email}, {phone}, last seen from {ip} in {location}."
        ),
        slots=(
            ("person", "Ivana Horvat", "PERSON"),
            ("email", "i.horvat@example.com", "EMAIL_ADDRESS"),
            ("phone", "555-010-6677", "PHONE_NUMBER"),
            ("ip", "203.0.113.77", "IP_ADDRESS"),
            ("location", "Zagreb", "LOCATION"),
        ),
        note="Five types in one line: partial detection shows up as partial recall.",
    ),
    _Template(
        id="dense-narrative",
        template=(
            "{first} escalated to {second} after the {location} node failed. The follow-up "
            "went to {email} and a copy was posted to ticket INC-44120."
        ),
        slots=(
            ("first", "Rowan Achterberg", "PERSON"),
            ("second", "Camila Restrepo", "PERSON"),
            ("location", "Porto", "LOCATION"),
            ("email", "followup@example.net", "EMAIL_ADDRESS"),
        ),
        note="Entities interleaved with a non-PII ticket identifier.",
    ),
    _Template(
        id="mixed-repeat-across-types",
        template=(
            "{person} opened the case; {repeat} closed it three days later after "
            "confirming the balance."
        ),
        slots=(
            ("person", "Yara Haddad", "PERSON"),
            ("repeat", "Yara Haddad", "PERSON"),
        ),
        note="Same person twice: both occurrences must be masked.",
    ),
    _Template(
        id="mixed-pii-and-nearmiss",
        template=(
            "{person} referenced build 10.2.3 and container AX-9910 while investigating "
            "the alert from {ip}."
        ),
        slots=(
            ("person", "Emil Dabrowski", "PERSON"),
            ("ip", "192.0.2.130", "IP_ADDRESS"),
        ),
        note="Real PII beside a version and a code that must not be flagged.",
    ),
    # --- filling the pre-registered quota --------------------------------------------
    # Added to reach the counts in CORPUS_PLAN, which was written before the corpus and is
    # a structural target only. No document here was chosen in response to a detector score.
    _Template(
        id="card-in-sentence",
        template="Settlement for card {card} cleared overnight without a chargeback.",
        slots=(("card", "6011111111111117", "CREDIT_CARD"),),
        note="Standard test card, Discover prefix.",
    ),
    _Template(
        id="email-in-parentheses",
        template="The approver ({email}) has not responded to the reminder.",
        slots=(("email", "approvals@example.net", "EMAIL_ADDRESS"),),
        note="Address enclosed in parentheses.",
    ),
    _Template(
        id="phone-in-parentheses-only",
        template="The after-hours line ({phone}) forwards to the regional desk.",
        slots=(("phone", "555-010-1188", "PHONE_NUMBER"),),
    ),
    _Template(
        id="ssn-sentence-initial",
        template="{ssn} was recorded twice in the intake system and later deduplicated.",
        slots=(("ssn", "678-90-1234", "US_SSN"),),
        note="Identifier at the very start of a sentence.",
    ),
)


def labeled_corpus() -> list[PIISample]:
    """The full synthetic corpus, offsets computed from the templates."""
    return [template.build() for template in _TEMPLATES]


def corpus_span_count() -> int:
    """Total labeled PII spans, for the report's corpus description."""
    return sum(len(sample.spans) for sample in labeled_corpus())
