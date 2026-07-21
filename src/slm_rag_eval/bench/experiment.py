"""Run the whole report experiment in one command.

Rental day should be execution, not improvisation. This drives the full matrix — several
SLM judges, a cloud baseline, a masked/unmasked ablation — and then every analysis, so that
one invocation leaves behind all of the report's evidence:

    report/<name>/
      environment.md / .json     Table 5.3, captured from the machine that ran it
      dataset.md                 Table 5.1, computed from the samples actually scored
      summary.md                 Tables 6.1–6.3, held-out and in-sample
      results.json               every number above, machine-readable
      reliability.png            Figure 6.1
      tradeoff.png               Figure 6.3
      roc_curves.png, score_distributions.png, latency_box.png
      errors/error_analysis.md   Table 6.4 with worked examples
      privacy/pii_detection.md   PII precision and recall
      rows/                      the raw JSONL and manifests behind all of it

Each judge lands in its own `--out` directory, because two judges resuming into one rows
file is refused by design. Failures do not abort the matrix: a judge that dies still leaves
its rows, and the run continues to the next one, since losing four hours of cloud results
because a local model crashed would be the expensive kind of mistake.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer

from slm_rag_eval.bench import environment as environment_module
from slm_rag_eval.bench.analyze import CostModel, analyze
from slm_rag_eval.bench.datasets import describe, load, render_stats
from slm_rag_eval.bench.resources import ResourceSampler, ResourceUsage, ollama_resident_models
from slm_rag_eval.bench.run import apply_privacy_mode, build_judge, output_paths, run_benchmark
from slm_rag_eval.core.config import Settings, get_settings

app = typer.Typer(help=__doc__, add_completion=False)


@dataclass
class Leg:
    """One judge run in the matrix."""

    name: str
    judge: str
    model: str
    privacy_mode: str
    status: str = "pending"
    detail: str = ""
    seconds: float = 0.0
    manifest: dict[str, Any] = field(default_factory=dict)
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    """Peak VRAM and memory while this leg ran — report R4, measured not remembered."""
    resident: list[dict[str, Any]] = field(default_factory=list)


def _settings_for(model: str, privacy_mode: str) -> Settings:
    """Base settings with this leg's model and privacy mode, re-validated not merged."""
    base = apply_privacy_mode(get_settings(), privacy_mode)
    return Settings.model_validate({**base.model_dump(), "model": model}, strict=False)


async def run_leg(
    leg: Leg,
    *,
    dataset: str,
    samples: list[Any],
    out_dir: Path,
    metrics: list[str],
    sampling: dict[str, Any],
) -> Leg:
    """Score one judge over the sample set, recording the outcome rather than raising."""
    started = time.perf_counter()
    settings = _settings_for(leg.model, leg.privacy_mode)
    try:
        client, judge_settings = build_judge(leg.judge, settings)
    except ValueError as exc:
        leg.status, leg.detail = "skipped", str(exc)
        return leg

    try:
        with ResourceSampler() as sampler:
            leg.manifest = await run_benchmark(
                samples,
                judge=client,
                dataset=dataset,
                judge_name=leg.judge,
                model=judge_settings.model,
                out_dir=out_dir,
                settings=judge_settings,
                metrics=metrics,
                sampling=sampling,
            )
            # Read while the model is still loaded; after the run it has been evicted.
            leg.resident = ollama_resident_models(judge_settings.base_url)
        leg.usage = sampler.usage
        written = leg.manifest.get("samples_written", 0)
        failures = leg.manifest.get("failure_count", 0)
        leg.status = "ok" if failures == 0 else "partial"
        leg.detail = f"{written} scored, {failures} failed"
    except Exception as exc:  # One dead judge must not cost the rest of the matrix.
        leg.status, leg.detail = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()
        leg.seconds = time.perf_counter() - started
    return leg


def plan_legs(models: list[str], *, cloud: bool, ablation_model: str | None) -> list[Leg]:
    """The matrix: every SLM masked, optionally one unmasked, optionally a cloud baseline."""
    legs = [
        Leg(name=f"slm-{model}", judge="slm", model=model, privacy_mode="mask")
        for model in models
    ]
    if ablation_model:
        legs.append(
            Leg(
                name=f"slm-{ablation_model}-nomask",
                judge="slm",
                model=ablation_model,
                privacy_mode="off",
            )
        )
    if cloud:
        legs.append(
            Leg(
                name="cloud",
                judge="cloud",
                model=os.environ.get("CLOUD_MODEL", "cloud"),
                privacy_mode="mask",
            )
        )
    return legs


def render_plan(legs: list[Leg]) -> str:
    """Human-readable matrix, printed before anything expensive starts."""
    lines = ["| Leg | Judge | Model | Privacy |", "|---|---|---|---|"]
    lines += [
        f"| {leg.name} | {leg.judge} | {leg.model} | {leg.privacy_mode} |" for leg in legs
    ]
    return "\n".join(lines)


def render_run_log(legs: list[Leg]) -> str:
    """What actually happened, including the legs that did not finish."""
    lines = [
        "| Leg | Status | Detail | Minutes | Peak VRAM | Peak system RAM |",
        "|---|---|---|---|---|---|",
    ]
    for leg in legs:
        vram = (
            max(leg.usage.peak_gpu_memory_mib.values(), default=0)
            if leg.usage.peak_gpu_memory_mib
            else 0
        )
        vram_text = f"{vram} MiB" if vram else "no GPU"
        ram = f"{leg.usage.peak_system_used_mib} MiB" if leg.usage.peak_system_used_mib else "n/a"
        lines.append(
            f"| {leg.name} | {leg.status} | {leg.detail} | {leg.seconds / 60:.1f} | "
            f"{vram_text} | {ram} |"
        )
    return "\n".join(lines)


def write_resource_report(legs: list[Leg], out: Path) -> Path:
    """Report R4: what each judge actually needed from the hardware."""
    payload = {
        leg.name: {
            "model": leg.model,
            "privacy_mode": leg.privacy_mode,
            "seconds": leg.seconds,
            "resident_models": leg.resident,
            **leg.usage.as_dict(),
        }
        for leg in legs
    }
    path = out / "resources.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@app.command()
def main(
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help="SLM judge to evaluate; repeat for several."),
    ] = None,
    dataset: Annotated[str, typer.Option(help="Dataset to score.")] = "ragtruth",
    split: Annotated[str | None, typer.Option(help="Official split, e.g. test.")] = "test",
    limit: Annotated[int | None, typer.Option(help="Samples to score per judge.")] = 150,
    seed: Annotated[
        int, typer.Option(help="Sampling seed, recorded in every manifest.")
    ] = 20260806,
    stratify: Annotated[
        bool, typer.Option("--stratify/--no-stratify", help="Keep the label/task balance.")
    ] = True,
    cloud: Annotated[
        bool, typer.Option("--cloud/--no-cloud", help="Include the cloud baseline leg.")
    ] = True,
    ablation_model: Annotated[
        str | None,
        typer.Option(help="Model to also run unmasked, for the privacy ablation."),
    ] = None,
    metric: Annotated[list[str] | None, typer.Option(help="Repeatable.")] = None,
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("report/experiment"),
    data_dir: Annotated[Path | None, typer.Option(help="Dataset cache directory.")] = None,
    include_rows: Annotated[
        list[Path] | None,
        typer.Option(
            "--include-rows",
            help="Existing rows JSONL to fold into the analysis; repeat. Used to bring a "
            "cloud baseline scored elsewhere into a run that only scores local models.",
        ),
    ] = None,
    cloud_input_cost_per_1m: Annotated[float, typer.Option()] = 0.0,
    cloud_output_cost_per_1m: Annotated[float, typer.Option()] = 0.0,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan and exit without scoring.")
    ] = False,
) -> None:
    """Run every judge, then every analysis, into one report directory."""
    import asyncio

    models = list(model or [])
    if not models:
        raise typer.BadParameter("Pass at least one --model.")
    metrics = list(metric) if metric else ["faithfulness"]
    legs = plan_legs(models, cloud=cloud, ablation_model=ablation_model)

    typer.echo(render_plan(legs))
    typer.echo("")
    if dry_run:
        typer.echo("dry run: nothing was scored.")
        return

    samples = load(dataset, limit, data_dir=data_dir, split=split, seed=seed, stratify=stratify)
    sampling = {"split": split, "seed": seed, "stratify": stratify, "limit": limit}
    out.mkdir(parents=True, exist_ok=True)
    rows_root = out / "rows"

    # Written first: if the run dies halfway, the machine it died on is still on record.
    captured = environment_module.capture()
    (out / "environment.md").write_text(
        environment_module.render(captured) + "\n", encoding="utf-8"
    )
    (out / "environment.json").write_text(json.dumps(asdict(captured), indent=2), encoding="utf-8")
    (out / "dataset.md").write_text(
        render_stats(describe(dataset, samples)) + "\n", encoding="utf-8"
    )

    for leg in legs:
        typer.echo(f"--- {leg.name} ({leg.model}, privacy={leg.privacy_mode}) ---")
        asyncio.run(
            run_leg(
                leg,
                dataset=dataset,
                samples=samples,
                out_dir=rows_root / leg.name,
                metrics=metrics,
                sampling=sampling,
            )
        )
        typer.echo(f"{leg.status}: {leg.detail} ({leg.seconds / 60:.1f} min)")

    rows_files = [
        output_paths(rows_root / leg.name, dataset, leg.judge)[0]
        for leg in legs
        if leg.status in {"ok", "partial"}
    ]
    rows_files = [path for path in rows_files if path.exists()]

    # Rows scored elsewhere join the analysis as their own series. A cloud judge costs the
    # same per token wherever it runs, and waiting on a remote API burns rented GPU hours
    # doing nothing, so it is worth scoring before the machine is hired.
    for extra in include_rows or []:
        if not extra.exists():
            raise typer.BadParameter(f"--include-rows {extra} does not exist")
        rows_files.append(extra)
        typer.echo(f"including external rows: {extra}")

    (out / "run_log.md").write_text(render_run_log(legs) + "\n", encoding="utf-8")
    write_resource_report(legs, out)

    if not rows_files:
        typer.echo("no judge produced rows; nothing to analyze.", err=True)
        raise typer.Exit(code=1)

    analyze(
        rows_files,
        out,
        costs=CostModel(
            cloud_input_per_1m=cloud_input_cost_per_1m,
            cloud_output_per_1m=cloud_output_cost_per_1m,
        ),
    )
    _run_error_analysis(rows_files, dataset, data_dir, out / "errors")
    _run_privacy_evaluation(out / "privacy")

    typer.echo("")
    typer.echo(render_run_log(legs))
    typer.echo("")
    typer.echo(f"report written to {out}")
    for artifact in sorted(path.name for path in out.iterdir()):
        typer.echo(f"  {artifact}")


def _run_error_analysis(
    rows_files: list[Path],
    dataset: str,
    data_dir: Path | None,
    out: Path,
) -> None:
    """Error analysis is a report section, not an optional extra — but never fatal."""
    from slm_rag_eval.bench import errors as errors_module

    try:
        frame = errors_module.assign_series(errors_module.load_rows(rows_files))
        reports = {
            str(series): errors_module.analyze_judge(frame[frame["series"] == series])
            for series in sorted(frame["series"].dropna().unique())
        }
        samples_by_id = {sample.id: sample for sample in load(dataset, data_dir=data_dir)}
        rows = errors_module._rows_with_buckets(frame, samples_by_id)  # noqa: SLF001
        cases = errors_module.collect_errors(rows, reports)
        totals = errors_module.bucket_totals(rows)
        out.mkdir(parents=True, exist_ok=True)
        (out / "error_analysis.md").write_text(
            errors_module.render_report(cases, totals) + "\n", encoding="utf-8"
        )
        typer.echo(f"error analysis: {len(cases)} misclassified samples -> {out}")
    except Exception as exc:
        typer.echo(f"error analysis skipped: {type(exc).__name__}: {exc}", err=True)


def _run_privacy_evaluation(out: Path) -> None:
    """PII detection quality needs no judge, so it runs even when every leg failed."""
    from slm_rag_eval.privacy.evaluation import (
        detection_misses,
        evaluate_detection,
        presidio_detector,
        render_report,
    )
    from slm_rag_eval.privacy.fixtures import labeled_corpus

    try:
        settings = get_settings()
        corpus = labeled_corpus()
        detector = presidio_detector(settings)
        results = evaluate_detection(corpus, detector)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pii_detection.md").write_text(
            render_report(
                results, corpus_size=len(corpus), misses=detection_misses(corpus, detector)
            )
            + "\n",
            encoding="utf-8",
        )
        overall = results["ALL"]
        typer.echo(
            f"privacy: PII recall {overall.recall:.3f}, precision {overall.precision:.3f} -> {out}"
        )
    except Exception as exc:
        typer.echo(f"privacy evaluation skipped: {type(exc).__name__}: {exc}", err=True)


if __name__ == "__main__":  # pragma: no cover
    app()
