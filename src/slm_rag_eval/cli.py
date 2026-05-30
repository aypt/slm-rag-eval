"""Command line interface (`rageval`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import typer

from slm_rag_eval import __version__
from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.core.schemas import EvalRequest, EvalResult
from slm_rag_eval.demo import (
    batch_row,
    format_verdict_table,
    score_summary,
    total_latency_ms,
    verdict_rows,
)
from slm_rag_eval.llm import build_client
from slm_rag_eval.llm.client import LLMClient
from slm_rag_eval.metrics.registry import evaluate, validate_metric_selection

app = typer.Typer(help="slm-rag-eval — privacy-preserving RAG evaluation with SLM judges.")


def judge_factory(settings: Settings) -> LLMClient:
    """Seam: tests replace this module attribute with one returning a FakeLLMClient."""
    return build_client(settings)


@app.callback()
def main() -> None:
    """Privacy-preserving RAG evaluation with SLM judges."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


def _load_request(
    json_file: Path | None,
    question: str | None,
    answer: str | None,
    contexts: list[str] | None,
) -> EvalRequest:
    if json_file is not None:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        return EvalRequest.model_validate(payload)
    if question is None or answer is None:
        raise typer.BadParameter("Provide --json FILE, or both --question and --answer.")
    return EvalRequest(question=question, answer=answer, contexts=list(contexts or []))


async def _evaluate_async(
    request: EvalRequest,
    judge: LLMClient,
    settings: Settings,
    metrics: list[str],
) -> EvalResult:
    return await evaluate(
        request,
        judge=judge,
        metrics=metrics,
        k=settings.k,
        strict=settings.strict,
        settings=settings,
    )


async def _aclose(judge: LLMClient) -> None:
    """Close a judge client that owns transport resources; fakes simply have no `aclose`."""
    aclose = getattr(judge, "aclose", None)
    if aclose is not None:
        await aclose()


def _evaluate(
    request: EvalRequest,
    judge: LLMClient,
    settings: Settings,
    metrics: list[str],
) -> EvalResult:
    async def once() -> EvalResult:
        try:
            return await _evaluate_async(request, judge, settings, metrics)
        finally:
            await _aclose(judge)

    return asyncio.run(once())


@app.command(name="eval")
def eval_command(
    question: Annotated[str | None, typer.Option(help="The original question.")] = None,
    answer: Annotated[str | None, typer.Option(help="The answer to evaluate.")] = None,
    context: Annotated[
        list[str] | None, typer.Option(help="Retrieved context; repeat for several.")
    ] = None,
    json_file: Annotated[
        Path | None, typer.Option("--json", help="Read an EvalRequest from this JSON file.")
    ] = None,
    metric: Annotated[
        list[str] | None, typer.Option(help="Repeatable; defaults to the configured metrics.")
    ] = None,
    fail_under: Annotated[
        float, typer.Option(help="Exit 1 when faithfulness is below this score.")
    ] = 0.0,
) -> None:
    """Evaluate one answer and print the claim table and scores."""
    settings = get_settings()
    request = _load_request(json_file, question, answer, context)
    metrics = list(metric) if metric else list(settings.enabled_metrics)

    started = perf_counter()
    result = _evaluate(request, judge_factory(settings), settings, metrics)
    wall_ms = (perf_counter() - started) * 1000

    typer.echo(format_verdict_table(verdict_rows(result)))
    typer.echo("")
    for name, score in score_summary(result).items():
        if name in metrics:
            typer.echo(f"{name}: {'n/a' if score is None else f'{score:.3f}'}")
    typer.echo(f"judge: {settings.model}   privacy: {settings.privacy_mode}")
    typer.echo(f"judge time: {total_latency_ms(result):.1f} ms   wall: {wall_ms:.1f} ms")

    if result.faithfulness is not None and result.faithfulness < fail_under:
        typer.echo(f"FAIL: faithfulness {result.faithfulness:.3f} < --fail-under {fail_under}")
        raise typer.Exit(code=1)


@app.command()
def batch(
    input_file: Annotated[Path, typer.Argument(help="JSONL of requests to evaluate.")],
    out: Annotated[Path, typer.Option(help="Where to write the result rows.")] = Path(
        "results.jsonl"
    ),
    metric: Annotated[
        list[str] | None, typer.Option(help="Repeatable; defaults to the configured metrics.")
    ] = None,
    dataset: Annotated[
        str | None, typer.Option(help="Dataset name for the rows; defaults to the file stem.")
    ] = None,
    judge_name: Annotated[
        str, typer.Option("--judge-name", help="Judge label recorded in each row.")
    ] = "cli",
) -> None:
    """Evaluate every row of a JSONL file, writing bench-shaped rows (without labels)."""
    settings = get_settings()
    metrics = list(metric) if metric else list(settings.enabled_metrics)
    try:
        validate_metric_selection(metrics)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    requests: list[tuple[str, EvalRequest]] = []
    with input_file.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            sample_id = str(payload.pop("id", index))
            requests.append((sample_id, EvalRequest.model_validate(payload)))

    async def evaluate_all() -> list[dict[str, Any]]:
        # One event loop for the whole batch. A fresh `asyncio.run()` per row closes the
        # loop the shared HTTP client's connection pool is bound to, so the second row
        # against a keep-alive backend — every real judge — died with "Event loop is
        # closed". Fakes never showed it: they hold no connection.
        judge = judge_factory(settings)
        try:
            collected: list[dict[str, Any]] = []
            for sample_id, request in requests:
                result = await _evaluate_async(request, judge, settings, metrics)
                collected.append(
                    batch_row(
                        sample_id,
                        result,
                        model=settings.model,
                        latency_ms=total_latency_ms(result),
                        dataset=dataset or input_file.stem,
                        judge=judge_name,
                    )
                )
            return collected
        finally:
            await _aclose(judge)

    rows = asyncio.run(evaluate_all())

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"wrote {len(rows)} rows to {out}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Start the API server (uvicorn) in this process."""
    import uvicorn

    uvicorn.run("slm_rag_eval.service.api:app", host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
