"""Benchmark runner: score a labeled dataset with one judge and write JSONL rows.

Two judges are supported. `slm` is the local judge from `Settings` (the point of the
project). `cloud` is any OpenAI-compatible API configured through CLOUD_* environment
variables and is intended for PUBLIC benchmark data only — never send private or production
data to it.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import typer

from slm_rag_eval.bench.datasets import LabeledSample, load
from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.llm import build_client
from slm_rag_eval.llm.client import LLMClient
from slm_rag_eval.metrics.registry import evaluate

JUDGES = ("slm", "cloud")

app = typer.Typer(help=__doc__, add_completion=False)


def output_paths(out_dir: Path, dataset: str, judge: str) -> tuple[Path, Path]:
    """Rows file and its manifest. Re-running the same combination resumes the same file."""
    stem = f"{dataset}_{judge}"
    return out_dir / f"{stem}.jsonl", out_dir / f"{stem}.manifest.json"


def completed_sample_ids(rows_path: Path) -> set[str]:
    """Sample ids already written, so a resumed run can skip them."""
    if not rows_path.exists():
        return set()
    done: set[str] = set()
    with rows_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                # A partially written final line must not abort the resume.
                continue
            sample_id = row.get("sample_id")
            if sample_id is not None:
                done.add(str(sample_id))
    return done


def build_judge(judge: str, settings: Settings) -> tuple[LLMClient, Settings]:
    """Return the client for `judge` plus the settings it was built from."""
    if judge == "slm":
        return build_client(settings), settings
    if judge == "cloud":
        cloud_settings = _cloud_settings(settings)
        return build_client(cloud_settings), cloud_settings
    raise ValueError(f"Unknown judge: {judge}. Available judges: {', '.join(JUDGES)}")


def _cloud_settings(settings: Settings) -> Settings:
    missing = [name for name in ("CLOUD_BASE_URL", "CLOUD_MODEL") if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "The cloud judge needs " + " and ".join(missing) + " in the environment. "
            "Use it for PUBLIC benchmark data only."
        )
    return settings.model_copy(
        update={
            "base_url": os.environ["CLOUD_BASE_URL"],
            "model": os.environ["CLOUD_MODEL"],
            "api_key": os.environ.get("CLOUD_API_KEY"),
        }
    )


def _token_usage(model_info: dict[str, Any]) -> dict[str, int]:
    """Sum per-metric token usage into one per-sample total."""
    totals: dict[str, int] = {}
    for metric_info in model_info.values():
        if not isinstance(metric_info, dict):
            continue
        usage = metric_info.get("token_usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


async def evaluate_sample(
    sample: LabeledSample,
    *,
    judge: LLMClient,
    dataset: str,
    judge_name: str,
    model: str,
    metrics: list[str],
    settings: Settings,
) -> dict[str, Any]:
    """Score one sample and shape it into the benchmark row schema."""
    started = perf_counter()
    result = await evaluate(
        EvalRequest(question=sample.question, answer=sample.answer, contexts=sample.contexts),
        judge=judge,
        metrics=metrics,
        k=settings.k,
        strict=settings.strict,
        settings=settings,
    )
    latency_ms = (perf_counter() - started) * 1000

    scores: dict[str, float | None] = {"faithfulness": result.faithfulness}
    if "relevance" in metrics:
        scores["relevance"] = result.relevance
    return {
        "sample_id": sample.id,
        "dataset": dataset,
        "judge": judge_name,
        "model": model,
        "label_hallucinated": sample.label_hallucinated,
        "scores": scores,
        "verdicts": [verdict.model_dump() for verdict in result.verdicts],
        "latency_ms": latency_ms,
        "usage": _token_usage(result.model_info),
    }


async def run_benchmark(
    samples: list[LabeledSample],
    *,
    judge: LLMClient,
    dataset: str,
    judge_name: str,
    model: str,
    out_dir: Path,
    settings: Settings,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate `samples`, appending rows and writing the run manifest. Returns the manifest."""
    selected = metrics if metrics is not None else list(settings.enabled_metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path, manifest_path = output_paths(out_dir, dataset, judge_name)

    already_done = completed_sample_ids(rows_path)
    written = 0
    failures: list[dict[str, str]] = []

    with rows_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            if sample.id in already_done:
                continue
            try:
                row = await evaluate_sample(
                    sample,
                    judge=judge,
                    dataset=dataset,
                    judge_name=judge_name,
                    model=model,
                    metrics=selected,
                    settings=settings,
                )
            except Exception as exc:
                # Fail soft: one bad sample must not end a 200-sample run.
                failures.append({"sample_id": sample.id, "error": f"{type(exc).__name__}: {exc}"})
                typer.echo(f"sample {sample.id} failed: {exc}", err=True)
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1

    manifest = {
        "dataset": dataset,
        "judge": judge_name,
        "model": model,
        "metrics": selected,
        "params": {
            "k": settings.k,
            "strict": settings.strict,
            "privacy_mode": settings.privacy_mode,
            "limit": len(samples),
        },
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "samples_seen": len(samples),
        "samples_written": written,
        "samples_skipped": len(already_done),
        "failure_count": len(failures),
        "failures": failures,
        "rows_path": str(rows_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def git_sha() -> str | None:
    """Current commit, or None outside a git checkout — recorded for reproducibility."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


@app.command()
def main(
    dataset: Annotated[str, typer.Option(help="Dataset name to score.")] = "ragtruth",
    judge: Annotated[
        str, typer.Option(help="slm (local) or cloud (PUBLIC benchmark data only).")
    ] = "slm",
    limit: Annotated[int | None, typer.Option(help="Score only the first N samples.")] = None,
    out: Annotated[Path, typer.Option(help="Output directory for JSONL rows.")] = Path("results"),
    privacy_mode: Annotated[
        str | None, typer.Option(help="mask or off; defaults to the configured setting.")
    ] = None,
    metric: Annotated[
        list[str] | None, typer.Option(help="Repeatable; defaults to the configured metrics.")
    ] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Dataset cache directory.")] = None,
) -> None:
    """Run one judge over one dataset and write results/<dataset>_<judge>.jsonl."""
    settings = get_settings()
    if privacy_mode is not None:
        settings = settings.model_copy(update={"privacy_mode": privacy_mode})

    samples = load(dataset, limit, data_dir=data_dir)
    client, judge_settings = build_judge(judge, settings)

    manifest = asyncio.run(
        run_benchmark(
            samples,
            judge=client,
            dataset=dataset,
            judge_name=judge,
            model=judge_settings.model,
            out_dir=out,
            settings=judge_settings,
            metrics=list(metric) if metric else None,
        )
    )
    typer.echo(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    app()
