"""Faithfulness scoring through atomic-claim extraction and context verification."""

from __future__ import annotations

import json
from collections import Counter
from time import perf_counter
from typing import Any

from pydantic import BaseModel, RootModel

from slm_rag_eval.core.schemas import ClaimVerdict, EvalRequest, EvalResult
from slm_rag_eval.llm.client import LLMClient, LLMResponse
from slm_rag_eval.llm.structured import generate_json

_BATCH_SIZE = 5

_EXTRACTION_INSTRUCTIONS = """You extract atomic factual claims from an answer.

Rules:
- Return only JSON matching {"claims": ["claim", ...]}.
- Include only claims stated in the answer; do not add background knowledge.
- Ignore opinions, recommendations, questions, and hedges such as "might" or "probably".
- Make each claim atomic. Split conjoined facts into separate claims.
- Copy numbers, dates, and names verbatim from the answer.
- Keep enough wording that each claim can be checked independently.

Example 1
Answer: "Ada Lovelace wrote notes in 1843 and translated an article by Luigi Menabrea."
Output: {"claims":["Ada Lovelace wrote notes in 1843.",
"Ada Lovelace translated an article by Luigi Menabrea."]}

Example 2
Answer: "I think the blue design is beautiful. The trial probably involved 80 people."
Output: {"claims":[]}
"""

_VERIFICATION_INSTRUCTIONS = """Judge whether each claim is supported by the provided context.

Rules:
- Use only the provided context. Do not use outside knowledge.
- Return a JSON array with exactly one item per claim, in the same index order.
- Each item must contain claim, verdict, and reason.
- Copy the corresponding claim exactly into claim.
- Use supported only when the context directly entails the whole claim.
- Use unsupported when the context contradicts the claim or supports a different value.
- Use uncertain when the context is missing or insufficient.
- Give one concise sentence for reason, grounded in the context.
"""


class Claims(BaseModel):
    """Atomic factual claims extracted from an answer."""

    claims: list[str]


class _ClaimVerdicts(RootModel[list[ClaimVerdict]]):
    """Schema wrapper for a batch of index-aligned verdicts."""


class _RecordingClient:
    """Transparent client wrapper that accumulates completion metadata."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = "unknown"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = await self._client.complete(
            messages,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt_tokens += response.usage.get("prompt_tokens", 0)
        self.completion_tokens += response.usage.get("completion_tokens", 0)
        self.model = response.model
        return response


def _extraction_messages(answer: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": _EXTRACTION_INSTRUCTIONS},
        {
            "role": "user",
            "content": f"Extract the factual claims from this answer:\n{json.dumps(answer)}",
        },
    ]


def _verification_messages(
    contexts: list[str], claims: list[str]
) -> list[dict[str, Any]]:
    context_payload = [{"index": index, "text": text} for index, text in enumerate(contexts)]
    claim_payload = [{"index": index, "claim": claim} for index, claim in enumerate(claims)]
    content = (
        "Context passages:\n"
        f"{json.dumps(context_payload, ensure_ascii=False)}\n\n"
        "Claims to verify:\n"
        f"{json.dumps(claim_payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": _VERIFICATION_INSTRUCTIONS},
        {"role": "user", "content": content},
    ]


async def _extract_claims(client: LLMClient, answer: str) -> list[str]:
    messages = _extraction_messages(answer)
    extracted = await generate_json(client, messages, Claims)
    if extracted.claims:
        return extracted.claims

    retry_messages = [
        *messages,
        {"role": "assistant", "content": extracted.model_dump_json()},
        {
            "role": "user",
            "content": (
                "No factual claims were returned. Re-read the answer once and return any explicit, "
                "checkable factual claims. Return an empty claims list only if there truly are "
                "none."
            ),
        },
    ]
    retried = await generate_json(client, retry_messages, Claims)
    return retried.claims


async def _verify_batch(
    client: LLMClient,
    contexts: list[str],
    claims: list[str],
) -> list[ClaimVerdict]:
    messages = _verification_messages(contexts, claims)
    response = await generate_json(client, messages, _ClaimVerdicts)
    verdicts = response.root
    if len(verdicts) != len(claims):
        retry_messages = [
            *messages,
            {"role": "assistant", "content": response.model_dump_json()},
            {
                "role": "user",
                "content": (
                    f"You returned {len(verdicts)} items for {len(claims)} claims. "
                    f"Return exactly {len(claims)} items, aligned with claim indexes "
                    f"0 through {len(claims) - 1}."
                ),
            },
        ]
        verdicts = (await generate_json(client, retry_messages, _ClaimVerdicts)).root

    if len(verdicts) != len(claims):
        raise ValueError(
            "Judge returned a verdict count that did not match the claim count after one retry"
        )

    return [
        ClaimVerdict(claim=claim, verdict=verdict.verdict, reason=verdict.reason)
        for claim, verdict in zip(claims, verdicts, strict=True)
    ]


async def _verify_claims(
    client: LLMClient,
    contexts: list[str],
    claims: list[str],
    k: int,
) -> list[ClaimVerdict]:
    runs: list[list[ClaimVerdict]] = []
    for _ in range(k):
        run: list[ClaimVerdict] = []
        for start in range(0, len(claims), _BATCH_SIZE):
            batch = claims[start : start + _BATCH_SIZE]
            run.extend(await _verify_batch(client, contexts, batch))
        runs.append(run)

    voted: list[ClaimVerdict] = []
    for index, claim in enumerate(claims):
        candidates = [run[index] for run in runs]
        counts = Counter(candidate.verdict for candidate in candidates)
        highest_count = max(counts.values())
        winners = [verdict for verdict, count in counts.items() if count == highest_count]
        verdict = winners[0] if len(winners) == 1 else "uncertain"

        matching_reason = next(
            (candidate.reason for candidate in candidates if candidate.verdict == verdict),
            "The verification runs disagreed about whether the context supports the claim.",
        )
        voted.append(ClaimVerdict(claim=claim, verdict=verdict, reason=matching_reason))
    return voted


def _score(verdicts: list[ClaimVerdict], *, strict: bool) -> float | None:
    supported = sum(verdict.verdict == "supported" for verdict in verdicts)
    if strict:
        return supported / len(verdicts)

    decided = sum(verdict.verdict != "uncertain" for verdict in verdicts)
    return supported / decided if decided else None


def _model_info(client: _RecordingClient, *, k: int, strict: bool) -> dict[str, object]:
    return {
        "model": client.model,
        "k": k,
        "strict": strict,
        "token_usage": {
            "prompt_tokens": client.prompt_tokens,
            "completion_tokens": client.completion_tokens,
            "total_tokens": client.prompt_tokens + client.completion_tokens,
        },
    }


def _empty_result(
    client: _RecordingClient,
    *,
    reason: str,
    k: int,
    strict: bool,
    extraction_ms: float,
) -> EvalResult:
    return EvalResult(
        faithfulness=None,
        verdicts=[ClaimVerdict(claim="", verdict="uncertain", reason=reason)],
        model_info=_model_info(client, k=k, strict=strict),
        timings={"claim_extraction_ms": extraction_ms, "verification_ms": 0.0},
    )


async def score_faithfulness(
    request: EvalRequest,
    judge: LLMClient,
    *,
    strict: bool = True,
    k: int = 1,
) -> EvalResult:
    """Score how many atomic answer claims are supported by the supplied contexts."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer greater than or equal to 1")

    client = _RecordingClient(judge)
    if not request.answer.strip():
        return _empty_result(
            client,
            reason="The answer is empty, so there are no factual claims to verify.",
            k=k,
            strict=strict,
            extraction_ms=0.0,
        )

    extraction_started = perf_counter()
    claims = await _extract_claims(client, request.answer)
    extraction_ms = (perf_counter() - extraction_started) * 1000
    if not claims:
        return _empty_result(
            client,
            reason="No factual claims were found after two extraction attempts.",
            k=k,
            strict=strict,
            extraction_ms=extraction_ms,
        )

    verification_started = perf_counter()
    verdicts = await _verify_claims(client, request.contexts, claims, k)
    verification_ms = (perf_counter() - verification_started) * 1000
    return EvalResult(
        faithfulness=_score(verdicts, strict=strict),
        verdicts=verdicts,
        model_info=_model_info(client, k=k, strict=strict),
        timings={
            "claim_extraction_ms": extraction_ms,
            "verification_ms": verification_ms,
        },
    )
