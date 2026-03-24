from __future__ import annotations

import pytest

from slm_rag_eval.core.schemas import EvalRequest, EvalResult
from slm_rag_eval.metrics.registry import evaluate
from tests.conftest import FakeLLMClient


def _request() -> EvalRequest:
    return EvalRequest(
        question="Which material shields the module?",
        answer="The module uses a ceramic shield.",
        contexts=["The module is protected by a ceramic shield."],
    )


async def test_registry_merges_both_metrics_into_complete_result(
    fake_llm: FakeLLMClient,
) -> None:
    fake_llm.push(
        '{"claims":["The module uses a ceramic shield."]}',
        '[{"claim":"The module uses a ceramic shield.","verdict":"supported",'
        '"reason":"The context identifies the same shield material."}]',
        '{"questions":["Which material shields the module?",'
        '"What kind of shield does the module use?",'
        '"What protects the module?"]}',
        '{"ratings":['
        '{"rating":2,"reason":"The questions have identical intent."},'
        '{"rating":2,"reason":"Both ask for the shield material."},'
        '{"rating":1,"reason":"This asks more broadly what provides protection."}'
        "]}",
    )

    result = await evaluate(
        _request(),
        judge=fake_llm,
        metrics=["faithfulness", "relevance"],
    )

    assert isinstance(result, EvalResult)
    assert result.faithfulness == 1.0
    assert result.relevance == pytest.approx(5 / 6)
    assert [verdict.verdict for verdict in result.verdicts] == ["supported"]
    assert set(result.timings) == {"faithfulness_ms", "relevance_ms"}
    assert all(latency >= 0 for latency in result.timings.values())
    assert set(result.model_info) == {"faithfulness", "relevance"}
    assert result.model_info["faithfulness"]["model"] == "fake"
    assert result.model_info["relevance"]["token_usage"]["total_tokens"] == 0
    assert len(fake_llm.calls) == 4


async def test_registry_defaults_to_faithfulness(fake_llm: FakeLLMClient) -> None:
    fake_llm.push(
        '{"claims":["The module uses a ceramic shield."]}',
        '[{"claim":"The module uses a ceramic shield.","verdict":"supported",'
        '"reason":"The context identifies the same shield material."}]',
    )

    result = await evaluate(_request(), judge=fake_llm)

    assert result.faithfulness == 1.0
    assert result.relevance is None
    assert set(result.model_info) == {"faithfulness"}
    assert len(fake_llm.calls) == 2


async def test_registry_rejects_unknown_metric_before_judge_call(
    fake_llm: FakeLLMClient,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"Unknown metric\(s\): fluency.*Available metrics: faithfulness, relevance",
    ):
        await evaluate(_request(), judge=fake_llm, metrics=["faithfulness", "fluency"])

    assert fake_llm.calls == []
