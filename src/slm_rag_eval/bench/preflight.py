"""Pre-run check: prove a real benchmark would succeed before paying for the hardware.

A 200-sample run against a rented GPU is expensive to discover a problem in. Every check
here fails in seconds for a reason that would otherwise surface an hour into a run, or —
worse — not surface at all and quietly shrink the denominator:

* the endpoint answers but the configured model was never pulled;
* the judge returns prose instead of JSON, so every sample burns three repair attempts;
* the token budget is too small for the longest sample, so long contexts truncate and the
  samples that fail are exactly the hard ones;
* the dataset is missing, unbalanced, or not the split that was intended;
* Presidio cannot load, so `privacy_mode=mask` fails on the first sample;
* the output directory already holds rows from a different configuration.

Run it with the same environment the real run will use:

    python -m slm_rag_eval.bench.preflight --dataset ragtruth --judge slm --sample-size 3
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated

import httpx
import typer

from slm_rag_eval.bench.datasets import LabeledSample, load
from slm_rag_eval.bench.run import (
    JUDGES,
    IncompatibleResumeError,
    apply_privacy_mode,
    build_judge,
    check_resume_compatibility,
    output_paths,
    run_environment,
    run_identity,
)
from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.llm.errors import JSONGenerationError, TruncatedResponseError
from slm_rag_eval.metrics.registry import evaluate, validate_metric_selection

app = typer.Typer(help=__doc__, add_completion=False)

OK = "ok"
WARN = "warn"
FAIL = "fail"

_MARKS = {OK: "PASS", WARN: "WARN", FAIL: "FAIL"}


@dataclass
class Check:
    """One preflight result."""

    name: str
    status: str
    detail: str

    def render(self) -> str:
        """One line, aligned, safe for a plain terminal and for pasting into the report."""
        return f"[{_MARKS[self.status]}] {self.name:<28} {self.detail}"


def check_configuration(settings: Settings, metrics: list[str]) -> Check:
    """The exact configuration the run will use, echoed so it can be pasted into the report."""
    try:
        validate_metric_selection(metrics)
    except ValueError as exc:
        return Check("configuration", FAIL, str(exc))
    return Check(
        "configuration",
        OK,
        f"model={settings.model} base_url={settings.base_url} metrics={','.join(metrics)} "
        f"k={settings.k} strict={settings.strict} privacy={settings.privacy_mode} "
        f"max_tokens={settings.max_tokens}",
    )


def check_endpoint(settings: Settings) -> tuple[Check, list[str]]:
    """Whether the judge endpoint answers, and which models it says it serves."""
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    url = f"{settings.base_url.rstrip('/')}/models"
    try:
        response = httpx.get(url, timeout=10.0, headers=headers)
    except httpx.HTTPError as exc:
        return Check("judge endpoint", FAIL, f"{url} is unreachable: {exc}"), []
    if response.status_code != 200:
        return (
            Check("judge endpoint", FAIL, f"{url} answered HTTP {response.status_code}"),
            [],
        )

    served: list[str] = []
    try:
        body = response.json()
        for entry in body.get("data", []) if isinstance(body, dict) else []:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                served.append(entry["id"])
    except ValueError:
        return Check("judge endpoint", WARN, f"{url} answered 200 but not with JSON"), []
    return Check("judge endpoint", OK, f"{url} answered 200, {len(served)} model(s)"), served


def check_model_available(settings: Settings, served: list[str]) -> Check:
    """A pulled model. This is the check that catches "compose started before the pull"."""
    if not served:
        return Check("model available", WARN, "the endpoint listed no models; cannot verify")
    if settings.model in served:
        return Check("model available", OK, f"{settings.model} is served")
    # Ollama reports `name:tag`; a bare name is a common and confusing mismatch.
    near = [name for name in served if name.split(":")[0] == settings.model.split(":")[0]]
    hint = f" Did you mean: {', '.join(sorted(near))}?" if near else ""
    return Check(
        "model available",
        FAIL,
        f"{settings.model} is not served (have: {', '.join(sorted(served)[:5])}).{hint}",
    )


def check_dataset(
    dataset: str,
    limit: int | None,
    data_dir: Path | None,
    *,
    split: str | None = None,
    seed: int | None = None,
    stratify: bool = False,
) -> tuple[Check, list[LabeledSample]]:
    """Dataset present, plus the label balance the report has to state.

    Takes the same sampling arguments the real run will use, so a misspelled split or an
    unavailable one fails here rather than after the judge is already warm.
    """
    try:
        samples = load(
            dataset, limit, data_dir=data_dir, split=split, seed=seed, stratify=stratify
        )
    except (FileNotFoundError, ValueError) as exc:
        return Check("dataset", FAIL, str(exc)), []
    if not samples:
        return Check("dataset", FAIL, f"{dataset} loaded zero samples"), []

    positives = sum(sample.label_hallucinated for sample in samples)
    share = positives / len(samples)
    status = OK if 0.1 <= share <= 0.9 else WARN
    detail = (
        f"{len(samples)} samples, {positives} hallucinated ({share:.0%}), "
        f"{len(samples) - positives} not"
    )
    if status is WARN:
        detail += " — a class this rare makes F1 unstable; say so in Limitations"
    return Check("dataset", status, detail), samples


def check_context_budget(samples: list[LabeledSample], settings: Settings) -> Check:
    """Rough headroom check: characters in, tokens out, against the configured budget.

    Deliberately crude (~4 characters per token). It is not a tokenizer; it is a smoke
    alarm for the case where the longest sample cannot possibly fit in the budget.
    """
    if not samples:
        return Check("token budget", WARN, "no samples to size")

    lengths = [
        len(sample.question) + len(sample.answer) + sum(len(c) for c in sample.contexts)
        for sample in samples
    ]
    longest = max(lengths)
    median = statistics.median(lengths)
    # Output is dominated by the verdict batch: roughly one claim plus a reason per sentence.
    estimated_output = max(256, len(samples and samples[0].answer) // 4)
    detail = (
        f"prompt chars median={int(median)} max={longest} "
        f"(~{longest // 4} tokens); completion budget={settings.max_tokens}"
    )
    if settings.max_tokens < estimated_output:
        return Check("token budget", WARN, f"{detail} — consider raising SLMEVAL_MAX_TOKENS")
    return Check("token budget", OK, detail)


def check_holdout_viability(samples: list[LabeledSample]) -> Check:
    """Whether this sample set can support a held-out threshold at all.

    The held-out F1 is the number the report quotes. It needs both classes present in both
    halves of the hash split; too few positives and the analysis reports n/a after the run
    has already been paid for.
    """
    if not samples:
        return Check("held-out split", WARN, "no samples to check")

    from slm_rag_eval.bench.analyze import holdout_fold

    halves: dict[int, list[bool]] = {0: [], 1: []}
    for sample in samples:
        halves[holdout_fold("preflight", sample.id)].append(sample.label_hallucinated)

    sizes = {fold: len(labels) for fold, labels in halves.items()}
    if any(len(set(labels)) < 2 for labels in halves.values()):
        return Check(
            "held-out split",
            FAIL,
            f"one half holds a single class (sizes {sizes[0]}/{sizes[1]}). The held-out "
            "table would be all n/a. Raise --limit or use --stratify.",
        )
    smallest = min(sum(labels) for labels in halves.values())
    status = OK if smallest >= 10 else WARN
    detail = f"halves {sizes[0]}/{sizes[1]}, fewest positives in a half: {smallest}"
    if status is WARN:
        detail += " — too few for a stable F1; say so in Limitations or raise --limit"
    return Check("held-out split", status, detail)


def check_privacy(settings: Settings) -> Check:
    """Presidio has to load now, not on the first sample of a paid run."""
    if settings.privacy_mode != "mask":
        return Check("privacy layer", WARN, "privacy_mode=off — masking is NOT active")
    try:
        from slm_rag_eval.metrics.registry import build_sanitizer

        sanitizer = build_sanitizer(settings)
        batch = sanitizer.sanitize(["Contact Jane Doe at jane.doe@example.com."])
    except (RuntimeError, ImportError, OSError) as exc:
        return Check("privacy layer", FAIL, f"the detector could not load: {exc}")
    if batch.texts[0] == "Contact Jane Doe at jane.doe@example.com.":
        return Check("privacy layer", FAIL, "the detector loaded but masked nothing")
    return Check(
        "privacy layer",
        OK,
        f"{len(batch.mapping)} entity/entities masked in the probe string, "
        f"entities={len(settings.privacy_entities)} threshold={settings.privacy_score_threshold}",
    )


def check_output_dir(
    out: Path,
    dataset: str,
    judge_name: str,
    settings: Settings,
    metrics: list[str],
    model: str,
) -> Check:
    """Catch an incompatible resume now rather than after the first sample is scored."""
    rows_path, manifest_path = output_paths(out, dataset, judge_name)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Check("output directory", FAIL, f"{out} is not writable: {exc}")

    identity = run_identity(dataset, judge_name, model, metrics, settings)
    try:
        previous = check_resume_compatibility(
            rows_path, manifest_path, identity, run_environment(settings)
        )
    except IncompatibleResumeError as exc:
        return Check("output directory", FAIL, str(exc))
    if previous is None:
        return Check("output directory", OK, f"{out} is writable and holds no prior rows")
    return Check(
        "output directory",
        WARN,
        f"{rows_path} already holds {previous.get('samples_written', '?')} compatible rows; "
        "the run will resume rather than start over",
    )


async def check_live_round_trip(
    samples: list[LabeledSample],
    settings: Settings,
    metrics: list[str],
    judge_name: str,
    sample_size: int,
) -> list[Check]:
    """The check that matters: score a few real samples with the real judge.

    Everything above is necessary; this is the one that proves sufficiency. It reports the
    per-sample latency the full run will be extrapolated from, so the operator can decide
    whether 200 samples fits the budget before renting anything.
    """
    if not samples:
        return [Check("live round trip", FAIL, "no samples to score")]

    client, judge_settings = build_judge(judge_name, settings)
    chosen = samples[:sample_size]
    latencies: list[float] = []
    tokens: list[int] = []
    failures: list[str] = []
    truncations = 0

    try:
        for sample in chosen:
            started = perf_counter()
            try:
                result = await evaluate(
                    EvalRequest(
                        question=sample.question,
                        answer=sample.answer,
                        contexts=sample.contexts,
                    ),
                    judge=client,
                    metrics=metrics,
                    k=judge_settings.k,
                    strict=judge_settings.strict,
                    settings=judge_settings,
                )
            except TruncatedResponseError as exc:
                truncations += 1
                failures.append(f"{sample.id}: {exc}")
                continue
            except (JSONGenerationError, ValueError) as exc:
                failures.append(f"{sample.id}: {type(exc).__name__}: {exc}")
                continue
            latencies.append((perf_counter() - started) * 1000)
            for info in result.model_info.values():
                if isinstance(info, dict):
                    usage = info.get("token_usage")
                    if isinstance(usage, dict):
                        tokens.append(int(usage.get("total_tokens", 0)))
    finally:
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()

    checks: list[Check] = []
    scored = len(latencies)

    if scored == 0:
        checks.append(
            Check("live round trip", FAIL, f"0/{len(chosen)} scored: {failures[0][:120]}")
        )
    else:
        mean_ms = statistics.mean(latencies)
        status = OK if scored == len(chosen) else WARN
        detail = f"{scored}/{len(chosen)} scored, mean {mean_ms:.0f} ms/sample"
        if failures:
            detail += f"; first failure: {failures[0][:90]}"
        checks.append(Check("live round trip", status, detail))

    # Reported whether or not anything scored: truncation is the failure with the cheapest
    # fix, so it must never be buried inside a generic "nothing scored" line.
    if truncations:
        checks.append(
            Check(
                "truncation",
                FAIL,
                f"{truncations}/{len(chosen)} samples hit the {settings.max_tokens}-token "
                "budget. Raise SLMEVAL_MAX_TOKENS before the real run.",
            )
        )

    if scored == 0:
        return checks

    if tokens:
        checks.append(
            Check("token usage", OK, f"mean {statistics.mean(tokens):.0f} tokens/sample")
        )

    # The number the rental decision is actually made on.
    for size in (100, 200):
        hours = mean_ms * size / 1000 / 3600
        checks.append(
            Check(
                f"projected {size}-sample run",
                OK if hours < 6 else WARN,
                f"~{hours:.2f} h at this latency (single judge, no concurrency)",
            )
        )
    return checks


@app.command()
def main(
    dataset: Annotated[str, typer.Option(help="Dataset the real run will use.")] = "ragtruth",
    judge: Annotated[str, typer.Option(help=f"One of: {', '.join(JUDGES)}.")] = "slm",
    sample_size: Annotated[
        int, typer.Option(help="How many real samples to score as a live probe.")
    ] = 3,
    limit: Annotated[
        int | None, typer.Option(help="Dataset limit the real run will use.")
    ] = None,
    split: Annotated[str | None, typer.Option(help="Split the real run will use.")] = None,
    seed: Annotated[int | None, typer.Option(help="Sampling seed the real run will use.")] = None,
    stratify: Annotated[
        bool, typer.Option("--stratify", help="Stratified draw, as the real run will use.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Output directory the real run will use.")] = Path(
        "results"
    ),
    metric: Annotated[list[str] | None, typer.Option(help="Repeatable.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Dataset cache directory.")] = None,
    skip_live: Annotated[
        bool, typer.Option("--skip-live", help="Skip the live probe (no judge calls).")
    ] = False,
) -> None:
    """Check everything a real benchmark run needs, then exit non-zero if anything failed."""
    settings = apply_privacy_mode(get_settings(), None)
    metrics = list(metric) if metric else list(settings.enabled_metrics)

    checks = [check_configuration(settings, metrics)]
    if checks[0].status == FAIL:
        _report(checks)
        raise typer.Exit(code=1)

    endpoint_check, served = check_endpoint(settings)
    checks.append(endpoint_check)
    if endpoint_check.status == OK:
        checks.append(check_model_available(settings, served))

    dataset_check, samples = check_dataset(
        dataset, limit, data_dir, split=split, seed=seed, stratify=stratify
    )
    checks.append(dataset_check)
    checks.append(check_context_budget(samples, settings))
    checks.append(check_holdout_viability(samples))
    checks.append(check_privacy(settings))
    checks.append(
        check_output_dir(out, dataset, judge, settings, metrics, settings.model)
    )

    if not skip_live and endpoint_check.status == OK and samples:
        checks.extend(
            asyncio.run(check_live_round_trip(samples, settings, metrics, judge, sample_size))
        )
    elif skip_live:
        checks.append(Check("live round trip", WARN, "skipped with --skip-live"))

    _report(checks)
    if any(check.status == FAIL for check in checks):
        raise typer.Exit(code=1)


def _report(checks: list[Check]) -> None:
    for check in checks:
        typer.echo(check.render())
    failed = sum(check.status == FAIL for check in checks)
    warned = sum(check.status == WARN for check in checks)
    typer.echo("")
    if failed:
        typer.echo(f"{failed} check(s) FAILED — do not start a paid run yet.")
    elif warned:
        typer.echo(f"All checks passed with {warned} warning(s). Read them before starting.")
    else:
        typer.echo("All checks passed. The real run should complete.")


if __name__ == "__main__":  # pragma: no cover
    app()
