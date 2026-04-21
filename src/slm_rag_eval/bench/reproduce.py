"""`make reproduce`: a 20-sample benchmark plus analysis, end to end.

Uses the configured Ollama judge when `SLMEVAL_BASE_URL` answers, and an explicitly
degenerate offline stub otherwise, so the command always exercises the whole path — loader →
masking → judge → rows → manifest → report — on any machine.

The stub is NOT an evaluator. It marks every claim `uncertain`, which in strict mode scores
0.0 for everything. That is deliberate: a smoke test must be impossible to mistake for a
result, and the report it writes is labeled as such.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx

from slm_rag_eval.bench.analyze import analyze
from slm_rag_eval.bench.datasets import LabeledSample, load
from slm_rag_eval.bench.run import build_judge, output_paths, run_benchmark
from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.llm.client import LLMResponse

SAMPLE_COUNT = 20
STUB_MODEL = "offline-stub"

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class OfflineStubJudge:
    """Schema-valid, deliberately uninformative judge for machines with no model running."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Answer whichever structured request was asked, without judging anything."""
        self.calls += 1
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        claims_to_verify = _claims_to_verify(prompt)

        payload: Any
        if claims_to_verify is not None:
            # Index-aligned with what was asked, so the batch contract is honored exactly.
            payload = [
                {"claim": claim, "verdict": "uncertain", "reason": "offline stub: not verified"}
                for claim in claims_to_verify
            ]
        elif "questions" in json.dumps(json_schema or {}):
            payload = {"questions": ["offline stub question"] * 3}
        elif "ratings" in json.dumps(json_schema or {}):
            payload = {"ratings": [{"rating": 0, "reason": "offline stub: not rated"}] * 3}
        else:
            payload = {"claims": _sentences(prompt)}
        return LLMResponse(text=json.dumps(payload), model=STUB_MODEL)


def _sentences(prompt: str) -> list[str]:
    tail = prompt.strip().splitlines()[-1] if prompt.strip() else ""
    parts = [part.strip() for part in _SENTENCE_END.split(tail) if part.strip()]
    return parts[:5] or ["offline stub claim"]


def _claims_to_verify(prompt: str) -> list[str] | None:
    """The claim batch a verification prompt carries, or None if this is not one."""
    marker = "Claims to verify:"
    if marker not in prompt:
        return None
    blob = prompt.split(marker, 1)[1].strip()
    try:
        items = json.loads(blob)
    except json.JSONDecodeError:
        return ["offline stub claim"]
    return [str(item.get("claim", "")) for item in items if isinstance(item, dict)]


def judge_is_reachable(settings: Settings, timeout_s: float = 2.0) -> bool:
    """True when the configured backend answers at all; any failure means 'use the stub'."""
    try:
        response = httpx.get(f"{settings.base_url.rstrip('/')}/models", timeout=timeout_s)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def reproduction_samples(settings: Settings, data_dir: Path | None = None) -> list[LabeledSample]:
    """Prefer a downloaded dataset; fall back to built-in synthetic samples."""
    for dataset in ("ragtruth", "halueval"):
        try:
            return load(dataset, SAMPLE_COUNT, data_dir=data_dir)
        except FileNotFoundError:
            continue
    return _builtin_samples()


def _builtin_samples() -> list[LabeledSample]:
    """Four invented samples so a fresh clone can reproduce without downloading anything."""
    facts = [
        ("The probe carried three instruments.", "The probe carried three instruments.", False),
        ("The probe carried seven instruments.", "The probe carried three instruments.", True),
        ("The launch happened in 2019.", "The launch happened in 2019.", False),
        ("The launch happened in 1998.", "The launch happened in 2019.", True),
    ]
    return [
        LabeledSample(
            id=f"builtin-{index}",
            question="What do the records say about the probe?",
            answer=answer,
            contexts=[context],
            label_hallucinated=hallucinated,
            meta={"dataset": "builtin-synthetic"},
        )
        for index, (answer, context, hallucinated) in enumerate(facts)
    ]


def reproduce(out_dir: Path = Path("report/repro"), data_dir: Path | None = None) -> dict[str, Any]:
    """Run the benchmark and the analysis into `out_dir`; returns the manifest."""
    settings = get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = out_dir / "rows"

    live = judge_is_reachable(settings)
    if live:
        judge, judge_settings = build_judge("slm", settings)
        judge_name, model = "slm", judge_settings.model
    else:
        judge, judge_settings = OfflineStubJudge(), settings
        judge_name, model = "stub", STUB_MODEL
        print(
            f"{settings.base_url} is not reachable — running the OFFLINE STUB judge.\n"
            "The report this writes is a smoke test, not an evaluation result."
        )

    samples = reproduction_samples(settings, data_dir)
    print(f"scoring {len(samples)} samples with the {judge_name} judge ({model})")

    manifest = asyncio.run(
        run_benchmark(
            samples,
            judge=judge,
            dataset="reproduce",
            judge_name=judge_name,
            model=model,
            out_dir=rows_dir,
            settings=judge_settings,
            metrics=["faithfulness"],
        )
    )
    rows_path, _ = output_paths(rows_dir, "reproduce", judge_name)
    outcome = analyze([rows_path], out_dir)

    print(json.dumps({k: str(v) for k, v in manifest.items() if k != "failures"}, indent=2))
    print(f"report: {outcome['summary_path']}")
    return manifest


def main() -> None:
    """Entry point for `make reproduce`."""
    reproduce()


if __name__ == "__main__":  # pragma: no cover
    main()
