"""Shared helpers for scripting judge responses.

Kept out of `conftest.py`, which AGENTS.md freezes. The verdict shape lives here in one
place because it is a wire contract: it changed once already when a top-level array turned
out not to be a valid structured-output root, and seven test files were building it by hand.
"""

from __future__ import annotations

import json

SUPPORTING_REASON = "The context agrees."


def verdict_batch(*claims: str, verdict: str = "supported", reason: str = SUPPORTING_REASON) -> str:
    """One verdict per claim, in the object-wrapped shape the judge schema requires."""
    return json.dumps(
        {
            "verdicts": [
                {"claim": claim, "verdict": verdict, "reason": reason} for claim in claims
            ]
        }
    )


def claim_list(*claims: str) -> str:
    """The extraction response for these claims."""
    return json.dumps({"claims": list(claims)})
