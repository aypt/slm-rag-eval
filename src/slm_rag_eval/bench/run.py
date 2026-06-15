"""Benchmark runner: score a labeled dataset with one judge and write JSONL rows.

Two judges are supported. `slm` is the local judge from `Settings` (the point of the
project). `cloud` is any OpenAI-compatible API configured through CLOUD_* environment
variables and is intended for PUBLIC benchmark data only — never send private or production
data to it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, NamedTuple

import typer

from slm_rag_eval.bench.datasets import LabeledSample, load
from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.llm import build_client
from slm_rag_eval.llm.client import LLMClient
from slm_rag_eval.metrics.registry import evaluate, validate_metric_selection

JUDGES = ("slm", "cloud")

app = typer.Typer(help=__doc__, add_completion=False)


def output_paths(out_dir: Path, dataset: str, judge: str) -> tuple[Path, Path]:
    """Rows file and its manifest. Re-running the same combination resumes the same file."""
    stem = f"{dataset}_{judge}"
    return out_dir / f"{stem}.jsonl", out_dir / f"{stem}.manifest.json"


class IncompatibleResumeError(ValueError):
    """Raised when an existing rows file was produced by a different run configuration."""


def run_identity(
    dataset: str,
    judge_name: str,
    model: str,
    metrics: list[str],
    settings: Settings,
) -> dict[str, Any]:
    """Everything that must match for two runs to belong in the same rows file.

    `limit` is excluded on purpose — extending a run to more samples is the whole point of
    resuming. The judge, the model, the metric set, and the scoring parameters are not:
    mixing them produces a file whose rows cannot all be attributed to one configuration.

    This is the core identity. `run_environment` carries the rest of what has to match;
    both are checked on resume and both feed `run_id`.
    """
    return {
        "dataset": dataset,
        "judge": judge_name,
        "model": model,
        "metrics": sorted(metrics),
        "k": settings.k,
        "strict": settings.strict,
        "privacy_mode": settings.privacy_mode,
    }


def run_environment(settings: Settings) -> dict[str, Any]:
    """The rest of the configuration that changes what a score means.

    `privacy_mode` alone does not describe the masking: the entity list and the confidence
    threshold decide what the judge was actually shown, so a mask-on/mask-off ablation is
    only interpretable when they are pinned too. `max_tokens` belongs here because it
    decides whether long samples truncate, and `base_url` because the same model name
    served by two endpoints is two different instruments.
    """
    return {
        "base_url": settings.base_url,
        "max_tokens": settings.max_tokens,
        "privacy_entities": sorted(settings.privacy_entities),
        "privacy_score_threshold": settings.privacy_score_threshold,
    }


def run_id(identity: dict[str, Any], environment: dict[str, Any]) -> str:
    """Short stable fingerprint of a run, used to keep incomparable rows from being pooled."""
    canonical = json.dumps(
        {"identity": identity, "environment": environment}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def check_resume_compatibility(
    rows_path: Path,
    manifest_path: Path,
    identity: dict[str, Any],
    environment: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Refuse to append to rows produced under a different configuration.

    Returns the previous manifest when resuming is safe, or None for a fresh run.
    """
    if not rows_path.exists() or rows_path.stat().st_size == 0:
        return None
    if not manifest_path.exists():
        raise IncompatibleResumeError(
            f"{rows_path} exists but {manifest_path} does not, so the configuration that "
            "produced those rows cannot be verified. Move them aside or use a new --out."
        )

    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_identity = previous.get("run_identity")
    if previous_identity is None:
        raise IncompatibleResumeError(
            f"{manifest_path} predates run-identity tracking, so a resume cannot be verified. "
            "Move the rows aside or use a new --out."
        )

    differences = [
        f"{key}: {previous_identity.get(key)!r} -> {value!r}"
        for key, value in identity.items()
        if previous_identity.get(key) != value
    ]
    if environment is not None:
        previous_environment = previous.get("run_environment")
        if previous_environment is None:
            raise IncompatibleResumeError(
                f"{manifest_path} predates run-environment tracking, so a resume cannot be "
                "verified. Move the rows aside or use a new --out."
            )
        differences += [
            f"{key}: {previous_environment.get(key)!r} -> {value!r}"
            for key, value in environment.items()
            if previous_environment.get(key) != value
        ]
    if differences:
        raise IncompatibleResumeError(
            "Refusing to resume into rows scored with a different configuration ("
            + "; ".join(differences)
            + f"). Use a new --out directory, or delete {rows_path}."
        )
    return previous


def read_rows(rows_path: Path) -> list[dict[str, Any]]:
    """Every parseable row in the file. A half-written final line is skipped, not fatal."""
    if not rows_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with rows_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                # A partially written final line must not abort the resume.
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def completed_sample_ids(rows_path: Path) -> set[str]:
    """Sample ids already written, so a resumed run can skip them."""
    return {
        str(row["sample_id"]) for row in read_rows(rows_path) if row.get("sample_id") is not None
    }


def drop_failed_rows(rows_path: Path) -> set[str]:
    """Remove failed rows from the file so a resume re-scores them. Returns their ids.

    A failed row is a real record and must not be silently overwritten, but a transient
    backend hiccup should not force a 200-sample re-run either. Dropping the rows here —
    and only when the operator asks — keeps exactly one row per sample.
    """
    rows = read_rows(rows_path)
    retryable = {str(row["sample_id"]) for row in rows if row.get("error") is not None}
    if not retryable:
        return set()
    kept = [row for row in rows if row.get("error") is None]
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8"
    )
    return retryable


class PrivacyMode(StrEnum):
    """CLI-visible privacy modes. An unrecognized value must never silently unmask."""

    mask = "mask"
    off = "off"


def apply_privacy_mode(settings: Settings, privacy_mode: str | None) -> Settings:
    """Override the configured privacy mode, re-validating instead of trusting the string.

    `model_copy(update=...)` skips validation, so a typo used to land in Settings unchecked
    and every non-`mask` value disables masking — a silent data leak. Re-validating turns a
    typo into an error.
    """
    if privacy_mode is None:
        return settings
    return Settings.model_validate(
        {**settings.model_dump(), "privacy_mode": privacy_mode},
        strict=False,
    )


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


def _returned_models(model_info: dict[str, Any]) -> set[str]:
    """Model names the backend actually reported, which can differ from the one configured.

    A routed or aliased cloud endpoint answers as whatever it picked, so recording only the
    configured string would misattribute the result to a model that never ran.
    """
    returned: set[str] = set()
    for metric_info in model_info.values():
        if isinstance(metric_info, dict):
            name = metric_info.get("model")
            if isinstance(name, str) and name and name != "unknown":
                returned.add(name)
    return returned


class SampleOutcome(NamedTuple):
    """One scored sample: the row to write, plus what the backend said it was."""

    row: dict[str, Any]
    models_returned: set[str]


async def evaluate_sample(
    sample: LabeledSample,
    *,
    judge: LLMClient,
    dataset: str,
    judge_name: str,
    model: str,
    metrics: list[str],
    settings: Settings,
) -> SampleOutcome:
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
    row = {
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
    return SampleOutcome(row=row, models_returned=_returned_models(result.model_info))


def failure_row(
    sample: LabeledSample,
    *,
    dataset: str,
    judge_name: str,
    model: str,
    metrics: list[str],
    error: str,
) -> dict[str, Any]:
    """A row for a sample the judge could not score, carrying null scores and the reason.

    Written so the analysis sees the sample as *unscored* rather than not seeing it at all.
    Coverage is part of the result: "193 of 200 scored" is a finding, and silently dropping
    the other seven turns a partial run into an apparently complete one.
    """
    scores: dict[str, float | None] = {"faithfulness": None}
    if "relevance" in metrics:
        scores["relevance"] = None
    return {
        "sample_id": sample.id,
        "dataset": dataset,
        "judge": judge_name,
        "model": model,
        "label_hallucinated": sample.label_hallucinated,
        "scores": scores,
        "verdicts": [],
        "latency_ms": None,
        "usage": {},
        "error": error,
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
    retry_failed: bool = False,
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate `samples`, appending rows and writing the run manifest. Returns the manifest."""
    selected = metrics if metrics is not None else list(settings.enabled_metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path, manifest_path = output_paths(out_dir, dataset, judge_name)

    validate_metric_selection(selected)
    identity = run_identity(dataset, judge_name, model, selected, settings)
    # `limit` stays out: extending a run to more samples is what resuming is for. Which
    # population the samples were drawn from is a different matter — resuming a test-split
    # run with training data would mix two populations into one file.
    draw = dict(sampling or {})
    draw.pop("limit", None)
    environment = {**run_environment(settings), "sampling": draw}
    previous = check_resume_compatibility(rows_path, manifest_path, identity, environment)
    retried = drop_failed_rows(rows_path) if retry_failed else set()
    already_done = completed_sample_ids(rows_path)
    written = 0
    failures: list[dict[str, str]] = []
    models_returned: set[str] = set()

    with rows_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            if sample.id in already_done:
                continue
            try:
                outcome = await evaluate_sample(
                    sample,
                    judge=judge,
                    dataset=dataset,
                    judge_name=judge_name,
                    model=model,
                    metrics=selected,
                    settings=settings,
                )
            except Exception as exc:
                # Fail soft: one bad sample must not end a 200-sample run. The failure is
                # still written as a row, because a sample that vanishes from the rows file
                # vanishes from the analysis too: a judge that fails selectively on hard or
                # hallucinated samples would otherwise look more accurate than it is. The
                # row carries null scores, so it counts as unscored, never as a result.
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"sample_id": sample.id, "error": error})
                typer.echo(f"sample {sample.id} failed: {exc}", err=True)
                row = failure_row(
                    sample,
                    dataset=dataset,
                    judge_name=judge_name,
                    model=model,
                    metrics=selected,
                    error=error,
                )
            else:
                row = outcome.row
                models_returned.update(outcome.models_returned)
                written += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    current_sha = git_sha()
    # Every commit whose rows are in this file: code can change between resumes even when
    # the configuration does not, and the report should say so.
    contributing_shas = list(previous.get("git_shas", []) if previous else [])
    if current_sha is not None and current_sha not in contributing_shas and written:
        contributing_shas.append(current_sha)

    manifest = {
        "dataset": dataset,
        "judge": judge_name,
        "model": model,
        "models_returned": sorted(models_returned),
        "metrics": selected,
        "run_identity": identity,
        "run_environment": environment,
        "run_id": run_id(identity, environment),
        "params": {
            "k": settings.k,
            "strict": settings.strict,
            "privacy_mode": settings.privacy_mode,
            "limit": len(samples),
        },
        "sampling": dict(sampling or {}),
        "label_balance": {
            "hallucinated": sum(sample.label_hallucinated for sample in samples),
            "not_hallucinated": sum(not sample.label_hallucinated for sample in samples),
        },
        "git_sha": current_sha,
        "git_shas": contributing_shas,
        "git_dirty": git_dirty(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "samples_seen": len(samples),
        "samples_written": written,
        "samples_skipped": len(already_done),
        "samples_retried": sorted(retried),
        "failure_count": len(failures),
        "failures": failures,
        "rows_path": str(rows_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def git_sha() -> str | None:
    """Current commit, or None outside a git checkout — recorded for reproducibility."""
    return _git(["rev-parse", "HEAD"]) or None


def git_dirty() -> bool | None:
    """Whether the checkout has uncommitted changes, or None outside a git checkout.

    A commit sha alone does not identify the code that produced a number: rows scored from
    a dirty tree are not reproducible from that sha, and the report has to be able to say so.
    """
    status = _git(["status", "--porcelain"])
    if status is None:
        return None
    return bool(status)


def _git(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


@app.command()
def main(
    dataset: Annotated[str, typer.Option(help="Dataset name to score.")] = "ragtruth",
    judge: Annotated[
        str, typer.Option(help="slm (local) or cloud (PUBLIC benchmark data only).")
    ] = "slm",
    limit: Annotated[int | None, typer.Option(help="Score only N samples.")] = None,
    split: Annotated[
        str | None, typer.Option(help="Official split to score, e.g. test. Fails if absent.")
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="Shuffle seed; recorded so the draw can be repeated.")
    ] = None,
    stratify: Annotated[
        bool, typer.Option("--stratify", help="Keep the label/task balance when limiting.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Output directory for JSONL rows.")] = Path("results"),
    privacy_mode: Annotated[
        PrivacyMode | None,
        typer.Option(help="mask or off; defaults to the configured setting."),
    ] = None,
    metric: Annotated[
        list[str] | None, typer.Option(help="Repeatable; defaults to the configured metrics.")
    ] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Dataset cache directory.")] = None,
    retry_failed: Annotated[
        bool, typer.Option("--retry-failed", help="Re-score samples that previously failed.")
    ] = False,
) -> None:
    """Run one judge over one dataset and write results/<dataset>_<judge>.jsonl."""
    settings = apply_privacy_mode(get_settings(), privacy_mode.value if privacy_mode else None)

    # Checked before anything expensive starts. Inside the run this same error would be
    # caught by the fail-soft loop once per sample, so a typo would burn a whole run and
    # still exit 0 with a manifest full of failures.
    selected = list(metric) if metric else list(settings.enabled_metrics)
    try:
        validate_metric_selection(selected)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    samples = load(
        dataset, limit, data_dir=data_dir, split=split, seed=seed, stratify=stratify
    )
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
            metrics=selected,
            retry_failed=retry_failed,
            sampling={"split": split, "seed": seed, "stratify": stratify, "limit": limit},
        )
    )
    typer.echo(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    app()
