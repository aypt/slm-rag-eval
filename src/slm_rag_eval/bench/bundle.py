"""Collect every report artifact into one directory, indexed and checksummed.

The rented machine is released at the end of the day and everything on it disappears. What
gets copied off has to be complete on the first attempt, because there is no second look —
so the bundle is one directory, with an index that says which report table each file
answers, and a checksum manifest so a truncated copy is detectable rather than silent.

The index is written from what is actually present. A file that was never produced is listed
as missing rather than omitted, because a gap the reader can see is recoverable and a gap
they cannot see is not.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help=__doc__, add_completion=False)


@dataclass(frozen=True)
class Artifact:
    """One expected output and the report item it supports."""

    relative_path: str
    report_item: str
    required: bool = True


EXPECTED: tuple[Artifact, ...] = (
    Artifact("environment.md", "Table 5.3 — experimental environment"),
    Artifact("environment.json", "Table 5.3 — machine-readable"),
    Artifact("dataset.md", "Table 5.1 — dataset statistics"),
    Artifact("dataset.lock.json", "Reproducibility — dataset identity"),
    Artifact("models/models.md", "Table 5.2 — judge model artifacts"),
    Artifact("models/models.json", "Table 5.2 — machine-readable"),
    Artifact("summary.md", "Tables 6.1–6.3 — detection quality, agreement, ablation"),
    Artifact("results.json", "All Chapter 6 numbers, machine-readable"),
    Artifact("reliability.png", "Figure 6.1 — reliability per judge"),
    Artifact("tradeoff.png", "Figure 6.3 — quality against latency and cost"),
    Artifact("roc_curves.png", "Figure 6.2 — ROC per judge"),
    Artifact("score_distributions.png", "Supporting figure — score distributions"),
    Artifact("latency_box.png", "R2 — latency distribution"),
    Artifact("run_log.md", "Coverage — what each leg did, including failures"),
    Artifact("resources.json", "R4 — peak VRAM and RAM per leg"),
    Artifact("errors/error_analysis.md", "Table 6.4 — failure modes with examples"),
    Artifact("privacy/pii_detection.md", "Q3 — PII detection precision and recall"),
    Artifact("privacy/pii_detection.json", "Q3 — machine-readable"),
    Artifact("deployment/deployment.md", "Q4 — Compose deployment transcript", required=False),
    Artifact("ablation/summary.md", "R5 — masking ablation", required=False),
    Artifact("make_check.txt", "Testing strategy — suite output at this revision", required=False),
)


def sha256_of(path: Path) -> str:
    """Streamed digest, so a large rows file does not have to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path) -> tuple[list[tuple[Artifact, Path]], list[Artifact]]:
    """Split the expected artifacts into those present and those missing."""
    present: list[tuple[Artifact, Path]] = []
    missing: list[Artifact] = []
    for artifact in EXPECTED:
        path = root / artifact.relative_path
        if path.exists():
            present.append((artifact, path))
        else:
            missing.append(artifact)
    return present, missing


def render_index(
    root: Path,
    present: list[tuple[Artifact, Path]],
    missing: list[Artifact],
    row_files: list[Path],
) -> str:
    """INDEX.md: which file answers which report item, and what is absent."""
    lines = [
        "# Experiment bundle",
        "",
        "Every artifact the report needs, produced by `slm_rag_eval.bench.campaign`.",
        "",
        "**Quote the held-out numbers** from `summary.md`. The in-sample table above them "
        "chooses its threshold on the rows it reports, so its F1 is an upper bound; both are "
        "printed so the difference is visible rather than hidden.",
        "",
        "## Artifacts",
        "",
        "| File | Answers | Present |",
        "|---|---|---|",
    ]
    for artifact, _ in present:
        lines.append(f"| `{artifact.relative_path}` | {artifact.report_item} | yes |")
    for artifact in missing:
        mark = "**NO — required**" if artifact.required else "no (optional)"
        lines.append(f"| `{artifact.relative_path}` | {artifact.report_item} | {mark} |")

    lines += [
        "",
        "## Raw rows",
        "",
        "Every number above derives from these. They are the only irreplaceable part of the "
        "bundle: figures and tables can be recomputed from rows, rows cannot be recomputed "
        "from anything.",
        "",
    ]
    lines += [f"- `{path.relative_to(root)}`" for path in sorted(row_files)]
    if not row_files:
        lines.append("- none found")

    lines += [
        "",
        "## Integrity",
        "",
        "`MANIFEST.sha256` lists a digest for every file. After copying the bundle, verify "
        "with:",
        "",
        "```bash",
        "cd <bundle> && sha256sum -c MANIFEST.sha256",
        "```",
        "",
        "A truncated copy is otherwise indistinguishable from a complete one until the "
        "numbers are needed.",
        "",
    ]
    return "\n".join(lines)


def write_manifest(root: Path) -> Path:
    """A `sha256sum -c` compatible manifest of every file in the bundle."""
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            entries.append(f"{sha256_of(path)}  {path.relative_to(root)}")
    manifest = root / "MANIFEST.sha256"
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest


def build(root: Path, *, archive: bool = True) -> tuple[list[Artifact], Path | None]:
    """Index and checksum an experiment directory; optionally tar it for transport."""
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    present, missing = collect(root)
    row_files = sorted(root.rglob("*.jsonl"))
    (root / "INDEX.md").write_text(
        render_index(root, present, missing, row_files), encoding="utf-8"
    )
    write_manifest(root)

    archive_path: Path | None = None
    if archive:
        archive_path = root.parent / f"{root.name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(root, arcname=root.name)
    return missing, archive_path


@app.command()
def main(
    root: Annotated[Path, typer.Argument(help="Experiment directory to bundle.")] = Path(
        "report/experiment"
    ),
    archive: Annotated[
        bool, typer.Option("--archive/--no-archive", help="Also write a .tar.gz beside it.")
    ] = True,
    extra: Annotated[
        list[Path] | None,
        typer.Option("--extra", help="Additional file or directory to copy in; repeat."),
    ] = None,
) -> None:
    """Index, checksum and archive everything the report needs."""
    for source in extra or []:
        if not source.exists():
            typer.echo(f"skipping missing --extra {source}", err=True)
            continue
        target = root / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    missing, archive_path = build(root, archive=archive)
    required_missing = [item for item in missing if item.required]

    typer.echo(f"bundle: {root}")
    typer.echo(f"index:  {root / 'INDEX.md'}")
    if archive_path is not None:
        size_mb = archive_path.stat().st_size / 1e6
        typer.echo(f"archive: {archive_path} ({size_mb:.1f} MB)  <- copy this off the host")
    if required_missing:
        typer.echo("")
        typer.echo("MISSING required artifacts:", err=True)
        for item in required_missing:
            typer.echo(f"  - {item.relative_path} ({item.report_item})", err=True)
        raise typer.Exit(code=1)
    typer.echo("")
    typer.echo("every required artifact is present.")


if __name__ == "__main__":  # pragma: no cover
    app()
