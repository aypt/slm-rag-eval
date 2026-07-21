"""Bring the Compose stack up, prove it works end to end, and write the transcript.

Report question Q4 asks whether the system deploys. The evidence for that is a transcript
of a real `docker compose up` on real hardware, which cannot be produced on a machine
without Docker — so this exists to be run once, unattended, on the rented host.

Every stage is timed and recorded separately, because "compose failed" is not a usable
finding. Whether the image built, whether Postgres became healthy, whether the model pull
completed, whether the API answered, and whether a submitted job actually finished are five
different problems with five different fixes.

Deliberately independent of the measurement legs: if the stack cannot start, the benchmark
numbers are still valid and already collected. This runs last for that reason.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated

import httpx
import typer

app = typer.Typer(help=__doc__, add_completion=False)

COMPOSE_FILES = ("docker-compose.yml",)
GPU_OVERLAY = "docker-compose.gpu.yml"


@dataclass
class Stage:
    """One step of the deployment check."""

    name: str
    ok: bool
    seconds: float
    detail: str = ""
    output: str = ""


@dataclass
class DeploymentResult:
    """The whole deployment attempt, stage by stage."""

    stages: list[Stage] = field(default_factory=list)
    job_id: str | None = None
    job_status: str | None = None

    @property
    def ok(self) -> bool:
        """True only when every stage succeeded."""
        return bool(self.stages) and all(stage.ok for stage in self.stages)


def _compose_command(files: tuple[str, ...], gpu: bool) -> list[str]:
    command = ["docker", "compose"]
    for name in files:
        command += ["-f", name]
    if gpu:
        command += ["-f", GPU_OVERLAY]
    return command


def _run(command: list[str], timeout_s: float, cwd: Path | None = None) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, cwd=cwd
        )
    except FileNotFoundError:
        return False, f"{command[0]} is not installed on this host"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_s:.0f}s"
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def _stage(name: str, command: list[str], timeout_s: float, cwd: Path | None = None) -> Stage:
    started = time.perf_counter()
    ok, output = _run(command, timeout_s, cwd)
    return Stage(
        name=name,
        ok=ok,
        seconds=time.perf_counter() - started,
        detail="" if ok else output.splitlines()[-1][:200] if output else "failed",
        output=output[-4000:],
    )


def wait_for_health(url: str, timeout_s: float, poll_s: float = 3.0) -> Stage:
    """Poll a health endpoint until it answers 200 or the budget runs out."""
    started = time.perf_counter()
    last = "no response"
    while time.perf_counter() - started < timeout_s:
        try:
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                return Stage(
                    name="api health",
                    ok=True,
                    seconds=time.perf_counter() - started,
                    detail=f"{url} answered 200",
                )
            last = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}"
        time.sleep(poll_s)
    return Stage(
        name="api health",
        ok=False,
        seconds=time.perf_counter() - started,
        detail=f"{url} never answered 200 within {timeout_s:.0f}s (last: {last})",
    )


def submit_and_poll(
    base: str, timeout_s: float, poll_s: float = 5.0
) -> tuple[Stage, str | None, str | None]:
    """Submit one evaluation through the API and wait for the worker to finish it.

    This is the stage that proves the whole stack, not just that containers are running:
    it exercises the API, the database, the queue, the worker and the model server.
    """
    started = time.perf_counter()
    payload = {
        "question": "How many instruments did the probe carry?",
        "answer": "The probe carried three instruments.",
        "contexts": ["The probe carried three instruments and launched in 2019."],
    }
    try:
        created = httpx.post(f"{base}/v1/evaluations", json=payload, timeout=30.0)
        created.raise_for_status()
        job_id = str(created.json()["id"])
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return (
            Stage(
                "job submit",
                False,
                time.perf_counter() - started,
                f"{type(exc).__name__}: {exc}",
            ),
            None,
            None,
        )

    status = "unknown"
    while time.perf_counter() - started < timeout_s:
        try:
            current = httpx.get(f"{base}/v1/evaluations/{job_id}", timeout=30.0).json()
            status = str(current.get("status"))
        except (httpx.HTTPError, ValueError):
            status = "unreachable"
        if status in {"done", "error"}:
            break
        time.sleep(poll_s)

    elapsed = time.perf_counter() - started
    return (
        Stage(
            name="job completes",
            ok=status == "done",
            seconds=elapsed,
            detail=f"job {job_id} finished as {status} after {elapsed:.0f}s",
        ),
        job_id,
        status,
    )


def render(result: DeploymentResult, *, gpu: bool) -> str:
    """The Q4 transcript: what ran, in what order, and how long each stage took."""
    lines = [
        "# Docker Compose deployment check",
        "",
        f"GPU overlay: {'applied' if gpu else 'not applied'}",
        "",
        "| Stage | Result | Seconds | Detail |",
        "|---|---|---|---|",
    ]
    for stage in result.stages:
        mark = "pass" if stage.ok else "FAIL"
        lines.append(f"| {stage.name} | {mark} | {stage.seconds:.1f} | {stage.detail} |")
    lines += [
        "",
        "The `job completes` stage is the one that matters: it exercises the API, the "
        "database, the queue, the worker and the model server in one request. Containers "
        "that are merely running prove none of that.",
        "",
    ]
    for stage in result.stages:
        if stage.output:
            lines += [f"## {stage.name}", "", "```", stage.output.strip(), "```", ""]
    return "\n".join(lines)


def run_deployment_check(
    *,
    out: Path,
    gpu: bool,
    api_base: str,
    build_timeout_s: float,
    start_timeout_s: float,
    job_timeout_s: float,
    keep_up: bool,
    project_dir: Path | None = None,
) -> DeploymentResult:
    """Build, start, verify and tear down the stack, recording every stage."""
    compose = _compose_command(COMPOSE_FILES, gpu)
    result = DeploymentResult()

    result.stages.append(_stage("docker available", ["docker", "version"], 60, project_dir))
    if not result.stages[-1].ok:
        return result

    result.stages.append(_stage("compose config", [*compose, "config"], 60, project_dir))
    if not result.stages[-1].ok:
        return result

    result.stages.append(_stage("build", [*compose, "build"], build_timeout_s, project_dir))
    if not result.stages[-1].ok:
        return result

    # `up --wait` blocks until dependencies report healthy, which includes ollama-init
    # completing its model pull. That wait is the point of the dependency change.
    result.stages.append(_stage("up (waits for model pull)", [*compose, "up", "-d", "--wait"],
                                start_timeout_s, project_dir))
    if not result.stages[-1].ok:
        result.stages.append(_stage("logs", [*compose, "logs", "--tail", "200"], 60, project_dir))
        if not keep_up:
            result.stages.append(_stage("down", [*compose, "down", "-v"], 180, project_dir))
        return result

    result.stages.append(wait_for_health(f"{api_base}/healthz", start_timeout_s))
    if result.stages[-1].ok:
        stage, job_id, status = submit_and_poll(api_base, job_timeout_s)
        result.stages.append(stage)
        result.job_id, result.job_status = job_id, status

    result.stages.append(
        _stage("logs", [*compose, "logs", "--tail", "200"], 60, project_dir)
    )
    if not keep_up:
        result.stages.append(_stage("down", [*compose, "down", "-v"], 180, project_dir))

    out.mkdir(parents=True, exist_ok=True)
    (out / "deployment.md").write_text(render(result, gpu=gpu) + "\n", encoding="utf-8")
    (out / "deployment.json").write_text(
        json.dumps(
            {
                "ok": result.ok,
                "job_id": result.job_id,
                "job_status": result.job_status,
                "gpu_overlay": gpu,
                "stages": [asdict(stage) for stage in result.stages],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Where to write the transcript.")] = Path(
        "report/deployment"
    ),
    gpu: Annotated[
        bool, typer.Option("--gpu/--no-gpu", help="Apply docker-compose.gpu.yml.")
    ] = True,
    api_base: Annotated[str, typer.Option(help="Where the API will listen.")] = (
        "http://localhost:8000"
    ),
    build_timeout_s: Annotated[float, typer.Option()] = 1800.0,
    start_timeout_s: Annotated[float, typer.Option(help="Covers the model pull.")] = 1800.0,
    job_timeout_s: Annotated[float, typer.Option()] = 600.0,
    keep_up: Annotated[
        bool, typer.Option("--keep-up", help="Leave the stack running afterwards.")
    ] = False,
) -> None:
    """Run the deployment check and exit non-zero if any stage failed."""
    result = run_deployment_check(
        out=out,
        gpu=gpu,
        api_base=api_base,
        build_timeout_s=build_timeout_s,
        start_timeout_s=start_timeout_s,
        job_timeout_s=job_timeout_s,
        keep_up=keep_up,
    )
    for stage in result.stages:
        typer.echo(f"[{'pass' if stage.ok else 'FAIL'}] {stage.name:<28} "
                   f"{stage.seconds:6.1f}s  {stage.detail}")
    typer.echo("")
    if result.ok:
        typer.echo(f"deployment verified; transcript in {out}")
        return
    typer.echo(f"deployment did NOT verify; see {out / 'deployment.md'}", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
