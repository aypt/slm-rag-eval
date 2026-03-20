"""Run the faithfulness pipeline without a network connection or real model."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.llm.client import LLMResponse
from slm_rag_eval.metrics.faithfulness import score_faithfulness


@dataclass
class FakeLLMClient:
    """Small scripted client that keeps this example network-free."""

    responses: list[str] = field(default_factory=list)

    def push(self, *responses: str) -> None:
        self.responses.extend(responses)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        del messages, json_schema, temperature, max_tokens
        if not self.responses:
            raise RuntimeError("The demo FakeLLMClient ran out of scripted responses")
        return LLMResponse(
            text=self.responses.pop(0),
            usage={"prompt_tokens": 120, "completion_tokens": 30},
            model="fake-demo-judge",
        )


async def main() -> None:
    judge = FakeLLMClient()
    judge.push(
        '{"claims":["The Aurora probe launched in 2032.",'
        '"The Aurora probe carried 4 instruments."]}',
        "["
        '{"claim":"The Aurora probe launched in 2032.","verdict":"supported",'
        '"reason":"The context states that Aurora launched in 2032."},'
        '{"claim":"The Aurora probe carried 4 instruments.","verdict":"unsupported",'
        '"reason":"The context states that Aurora carried 3 instruments."}'
        "]",
    )
    request = EvalRequest(
        question="What is known about the Aurora probe?",
        answer="The Aurora probe launched in 2032 and carried 4 instruments.",
        contexts=["The Aurora probe launched in 2032 carrying 3 scientific instruments."],
    )

    result = await score_faithfulness(request, judge)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
