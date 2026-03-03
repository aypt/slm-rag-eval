from __future__ import annotations

import pytest
from pydantic import ValidationError

from slm_rag_eval.core.schemas import ClaimVerdict, EvalRequest, EvalResult


def test_eval_request_roundtrip() -> None:
    req = EvalRequest(question="q", answer="a", contexts=["c1", "c2"])
    assert req.model_dump() == {"question": "q", "answer": "a", "contexts": ["c1", "c2"]}


def test_claim_verdict_rejects_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        ClaimVerdict(claim="x", verdict="maybe", reason="r")  # type: ignore[arg-type]


def test_eval_result_defaults() -> None:
    res = EvalResult()
    assert res.faithfulness is None
    assert res.verdicts == []
    assert res.timings == {}
