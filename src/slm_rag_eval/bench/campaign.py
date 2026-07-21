"""One command that runs the entire rented-machine day, with a gate before each paid step.

Written to be driven by an agent with no context beyond `docs/RUNBOOK.md`. Every step
prints what it is doing, checks a condition, and either continues or stops with a reason.
Steps that cost money come after the steps that would have caught a problem for free.

Order is deliberate:

  0  verify the code and the data are what we think they are — free, seconds
  1  freeze the model artifacts — free, needs the models pulled
  2  preflight every model — one real sample each, catches truncation and misconfiguration
  3  the primary matrix — the expensive step, and the only irreplaceable output
  4  the masking ablation — cheaper, and depends on step 3 to choose a model
  5  PII detection — needs no judge at all
  6  Compose deployment — independent of every number above, so a failure costs nothing
  7  bundle — index and checksum what to copy off the host

Steps 5 and 6 never block: PII needs no GPU, and a stack that will not start does not
invalidate measurements already taken.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help=__doc__, add_completion=False)


@dataclass
class Step:
    """One campaign step and what became of it."""

    name: str
    ok: bool = False
    skipped: bool = False
    seconds: float = 0.0
    detail: str = ""
    blocking: bool = True


@dataclass
class Campaign:
    """The whole run."""

    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> Step:
        """Record a step and echo its outcome."""
        self.steps.append(step)
        mark = "SKIP" if step.skipped else ("ok" if step.ok else "FAIL")
        typer.echo(f"[{mark}] {step.name}  ({step.seconds:.0f}s)  {step.detail}")
        return step

    def render(self) -> str:
        """The run log, including anything skipped or failed."""
        lines = ["| Step | Result | Seconds | Detail |", "|---|---|---|---|"]
        for step in self.steps:
            mark = "skipped" if step.skipped else ("ok" if step.ok else "FAIL")
            lines.append(f"| {step.name} | {mark} | {step.seconds:.0f} | {step.detail} |")
        return "\n".join(lines)


def _run(
    command: list[str], timeout_s: float = 3600.0, env: dict[str, str] | None = None
) -> tuple[bool, str]:
    """Run a subcommand, returning success and the tail of its output."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, **(env or {})},
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_s:.0f}s"
    except OSError as exc:
        return False, str(exc)
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def _python() -> list[str]:
    return [sys.executable, "-m"]


def _timed(
    name: str,
    command: list[str],
    timeout_s: float,
    *,
    blocking: bool = True,
    env: dict[str, str] | None = None,
) -> Step:
    started = time.perf_counter()
    ok, output = _run(command, timeout_s, env)
    tail = output.splitlines()[-1][:180] if output else ""
    return Step(
        name=name,
        ok=ok,
        seconds=time.perf_counter() - started,
        detail=tail if not ok else "",
        blocking=blocking,
    )


@app.command()
def main(
    model: Annotated[
        list[str] | None, typer.Option("--model", help="Local SLM tag to evaluate; repeat.")
    ] = None,
    dataset: Annotated[str, typer.Option()] = "ragtruth",
    split: Annotated[str, typer.Option()] = "test",
    limit: Annotated[int, typer.Option(help="Samples per judge in the primary matrix.")] = 150,
    ablation_limit: Annotated[int, typer.Option(help="Samples in the masking ablation.")] = 100,
    seed: Annotated[int, typer.Option()] = 20260806,
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("report/experiment"),
    base_url: Annotated[str, typer.Option()] = "http://localhost:11434/v1",
    include_rows: Annotated[
        list[Path] | None,
        typer.Option("--include-rows", help="Cloud rows scored elsewhere; repeat."),
    ] = None,
    lock: Annotated[Path, typer.Option(help="Dataset lock to verify against.")] = Path(
        "dataset.lock.json"
    ),
    cloud: Annotated[
        bool, typer.Option("--cloud/--no-cloud", help="Also score a cloud judge here.")
    ] = False,
    cloud_input_cost_per_1m: Annotated[float, typer.Option()] = 0.0,
    cloud_output_cost_per_1m: Annotated[float, typer.Option()] = 0.0,
    skip_deployment: Annotated[
        bool, typer.Option("--skip-deployment", help="Do not run the Compose check.")
    ] = False,
    preflight_samples: Annotated[int, typer.Option()] = 2,
) -> None:
    """Run every step of the experiment day and bundle the result."""
    models = list(model or [])
    if not models:
        raise typer.BadParameter("Pass at least one --model.")

    out.mkdir(parents=True, exist_ok=True)
    campaign = Campaign()
    sampling = [
        "--dataset", dataset, "--split", split, "--seed", str(seed), "--stratify",
    ]

    # --- 0. the code and the data are what we think they are ------------------------
    started = time.perf_counter()
    check_ok, check_output = _run(["make", "check"], 1800)
    # Saved either way: a failing suite is the reason the campaign stopped, and the report
    # needs the passing output as its testing-strategy evidence.
    (out / "make_check.txt").write_text(check_output, encoding="utf-8")
    campaign.add(
        Step(
            name="make check",
            ok=check_ok,
            seconds=time.perf_counter() - started,
            detail="" if check_ok else check_output.splitlines()[-1][:180],
        )
    )
    if not check_ok:
        typer.echo("STOP: the test suite does not pass on this checkout.", err=True)
        _finish(campaign, out)
        raise typer.Exit(code=1)

    if lock.exists():
        step = campaign.add(
            _timed(
                "dataset lock",
                [*_python(), "slm_rag_eval.bench.dataset_lock", "verify", str(lock)],
                600,
            )
        )
        if not step.ok:
            typer.echo("STOP: this machine's dataset differs from the locked one.", err=True)
            _finish(campaign, out)
            raise typer.Exit(code=1)
        shutil.copy2(lock, out / "dataset.lock.json")
    else:
        campaign.add(Step("dataset lock", ok=True, skipped=True, detail=f"{lock} not present"))

    # --- 1. freeze the artifacts ------------------------------------------------------
    freeze_command = [*_python(), "slm_rag_eval.bench.models", "--base-url", base_url,
                      "--out", str(out / "models")]
    for tag in models:
        freeze_command += ["--model", tag]
    campaign.add(_timed("freeze model artifacts", freeze_command, 300, blocking=False))

    # --- 2. preflight every model -----------------------------------------------------
    for tag in models:
        command = [
            *_python(), "slm_rag_eval.bench.preflight", "--judge", "slm",
            *sampling, "--limit", str(limit),
            "--sample-size", str(preflight_samples),
            "--out", str(out / "rows" / _slug(tag)),
        ]
        # The model is passed through the child's environment rather than a flag, because
        # preflight reads it from Settings exactly as the real run will.
        step = campaign.add(
            _timed(f"preflight {tag}", command, 3600, env={"SLMEVAL_MODEL": tag})
        )
        if not step.ok:
            typer.echo(
                f"STOP: preflight failed for {tag}. Fix it before spending run time.",
                err=True,
            )
            _finish(campaign, out)
            raise typer.Exit(code=1)

    # --- 3. the primary matrix --------------------------------------------------------
    experiment = [
        *_python(), "slm_rag_eval.bench.experiment",
        *sampling, "--limit", str(limit), "--out", str(out),
        "--cloud-input-cost-per-1m", str(cloud_input_cost_per_1m),
        "--cloud-output-cost-per-1m", str(cloud_output_cost_per_1m),
        "--cloud" if cloud else "--no-cloud",
    ]
    for tag in models:
        experiment += ["--model", tag]
    for path in include_rows or []:
        experiment += ["--include-rows", str(path)]
    step = campaign.add(_timed("primary matrix", experiment, 86400))
    if not step.ok:
        typer.echo("STOP: the primary matrix did not complete. Rows so far are kept.", err=True)
        _finish(campaign, out)
        raise typer.Exit(code=1)

    # --- 4. masking ablation, on the best model by held-out F1 ------------------------
    best = _best_model(out / "results.json", models)
    if best is None:
        campaign.add(
            Step("masking ablation", ok=True, skipped=True,
                 detail="no held-out F1 available to choose a model")
        )
    else:
        ablation = [
            *_python(), "slm_rag_eval.bench.experiment",
            "--model", best, "--ablation-model", best, "--no-cloud",
            *sampling, "--limit", str(ablation_limit),
            "--out", str(out / "ablation"),
        ]
        campaign.add(_timed(f"masking ablation ({best})", ablation, 86400, blocking=False))

    # --- 5. PII detection: no judge needed --------------------------------------------
    campaign.add(
        _timed(
            "PII detection",
            [*_python(), "slm_rag_eval.privacy.evaluation", "--out", str(out / "privacy")],
            1800,
            blocking=False,
        )
    )

    # --- 6. deployment: independent of every number above ------------------------------
    if skip_deployment:
        campaign.add(Step("compose deployment", ok=True, skipped=True, detail="--skip-deployment"))
    else:
        campaign.add(
            _timed(
                "compose deployment",
                [*_python(), "slm_rag_eval.bench.deployment", "--out", str(out / "deployment")],
                7200,
                blocking=False,
            )
        )

    # --- 7. bundle ---------------------------------------------------------------------
    _finish(campaign, out)
    step = campaign.add(
        _timed("bundle", [*_python(), "slm_rag_eval.bench.bundle", str(out)], 1800,
               blocking=False)
    )
    typer.echo("")
    typer.echo(campaign.render())
    typer.echo("")
    failed = [s for s in campaign.steps if not s.ok and not s.skipped]
    if failed:
        typer.echo(f"{len(failed)} step(s) failed; see {out / 'campaign_log.md'}", err=True)
    typer.echo(f"COPY THIS OFF THE HOST: {out.parent / (out.name + '.tar.gz')}")
    typer.echo(f"  and read {out / 'INDEX.md'} for what each file answers.")


def _slug(tag: str) -> str:
    return tag.replace("/", "-").replace(":", "-")


def _best_model(results_path: Path, models: list[str]) -> str | None:
    """The model with the highest held-out F1, which is the ablation's subject.

    Selected from the recorded results rather than by eye, and recorded in the log, so the
    choice is auditable rather than a decision someone made at the terminal.
    """
    if not results_path.exists():
        return None
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    best_tag, best_f1 = None, -1.0
    for entry in payload.get("judges", {}).values():
        held = entry.get("held_out")
        if not held:
            continue
        model_name = str(entry.get("model", ""))
        if model_name not in models:
            continue
        f1 = float(held.get("f1", -1.0))
        if f1 > best_f1:
            best_tag, best_f1 = model_name, f1
    return best_tag


def _finish(campaign: Campaign, out: Path) -> None:
    (out / "campaign_log.md").write_text(
        "# Campaign log\n\n" + campaign.render() + "\n", encoding="utf-8"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
