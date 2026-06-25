"""Capture the machine and software an experiment ran on (report Table 5.3, and R4).

Reproducibility claims rest on this: "qwen2.5:7b-instruct at 42 ms/sample" means nothing
without the hardware it was measured on. Collected automatically because a hand-written
environment table is the first thing to go stale when the experiment moves to a rented box.

Everything degrades gracefully — no GPU, no `nvidia-smi`, no git — because the same command
has to work on a laptop and on the rental.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help=__doc__, add_completion=False)

_TRACKED_PACKAGES = (
    "slm-rag-eval",
    "httpx",
    "pydantic",
    "presidio-analyzer",
    "spacy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "fastapi",
    "sqlalchemy",
)


@dataclass
class Environment:
    """Hardware and software identification for one experimental run."""

    python_version: str
    platform: str
    processor: str
    cpu_count: int | None
    total_ram_gb: float | None
    gpus: list[str] = field(default_factory=list)
    packages: dict[str, str] = field(default_factory=dict)
    git_sha: str | None = None
    git_dirty: bool | None = None


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def detect_gpus() -> list[str]:
    """GPU names and memory via nvidia-smi; empty on a CPU-only machine."""
    output = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def total_ram_gb() -> float | None:
    """Physical memory in GB, read from /proc where it exists."""
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def package_versions(names: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of the packages whose behaviour could move a number."""
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def capture() -> Environment:
    """Everything the methodology chapter needs about where the numbers came from."""
    from slm_rag_eval.bench.run import git_dirty, git_sha

    return Environment(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        processor=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
        total_ram_gb=total_ram_gb(),
        gpus=detect_gpus(),
        packages=package_versions(),
        git_sha=git_sha(),
        git_dirty=git_dirty(),
    )


def render(environment: Environment) -> str:
    """Report Table 5.3 as Markdown."""
    gpu = "; ".join(environment.gpus) if environment.gpus else "none detected (CPU only)"
    rows = [
        ("Python", environment.python_version),
        ("Operating system", environment.platform),
        ("Processor", environment.processor),
        ("Logical CPUs", str(environment.cpu_count) if environment.cpu_count else "unknown"),
        (
            "System RAM",
            f"{environment.total_ram_gb} GB" if environment.total_ram_gb else "unknown",
        ),
        ("GPU", gpu),
        (
            "Commit",
            f"{environment.git_sha}{' (dirty tree)' if environment.git_dirty else ''}"
            if environment.git_sha
            else "not a git checkout",
        ),
    ]
    lines = ["### Experimental environment", "", "| Component | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines += ["", "| Package | Version |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in sorted(environment.packages.items())]
    if not environment.gpus:
        lines += [
            "",
            "> No GPU was detected. Latency and throughput measured here reflect CPU "
            "inference and must not be compared with GPU figures.",
        ]
    return "\n".join(lines)


@app.command()
def main(
    out: Annotated[
        Path | None, typer.Option(help="Write the table and JSON here instead of stdout.")
    ] = None,
) -> None:
    """Print (or write) the experimental environment table."""
    environment = capture()
    table = render(environment)
    if out is None:
        typer.echo(table)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.md").write_text(table + "\n", encoding="utf-8")
    (out / "environment.json").write_text(
        json.dumps(asdict(environment), indent=2), encoding="utf-8"
    )
    typer.echo(table)
    typer.echo("")
    typer.echo(f"wrote {out / 'environment.md'}")


if __name__ == "__main__":  # pragma: no cover
    app()
