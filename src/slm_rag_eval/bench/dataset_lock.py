"""Pin the dataset bytes so a second machine can prove it has the same data.

The downloaders fetch from a mutable `main` branch. Two machines that run
`download_ragtruth.py` a week apart can therefore hold different corpora while every
manifest, seed and stratification setting looks identical — and the report would claim a
reproducibility it does not have.

The lock records a digest of each cached file plus the derived population: sample count,
label balance, and the identities the configured draw selects. Verifying it turns "we think
it is the same dataset" into a checkable statement, and it catches the likelier problem too
— a copy that arrived truncated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer

from slm_rag_eval.bench.datasets import DEFAULT_DATA_DIR, describe, load

app = typer.Typer(help=__doc__, add_completion=False)

DEFAULT_LOCK_PATH = Path("dataset.lock.json")

_DATASET_FILES = {
    "ragtruth": ("ragtruth/response.jsonl", "ragtruth/source_info.jsonl"),
    "halueval": ("halueval/qa_data.json",),
}


@dataclass
class FileDigest:
    """One cached dataset file, identified by content rather than by name and date."""

    path: str
    sha256: str
    bytes: int
    lines: int


@dataclass
class DatasetLock:
    """Everything needed to prove two machines are scoring the same population."""

    dataset: str
    files: list[FileDigest]
    total_samples: int
    sampling: dict[str, Any]
    drawn_samples: int
    drawn_hallucinated: int
    drawn_sample_ids_sha256: str
    """Digest of the ordered drawn ids: the strongest single check that the draw matches."""
    task_types: dict[str, int] = field(default_factory=dict)


def digest_file(path: Path) -> FileDigest:
    """Stream the file so a multi-gigabyte corpus does not have to fit in memory."""
    sha = hashlib.sha256()
    size = 0
    lines = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
            size += len(chunk)
            lines += chunk.count(b"\n")
    return FileDigest(path=path.name, sha256=sha.hexdigest(), bytes=size, lines=lines)


def build_lock(
    dataset: str,
    *,
    data_dir: Path | None = None,
    limit: int | None = None,
    split: str | None = None,
    seed: int | None = None,
    stratify: bool = False,
) -> DatasetLock:
    """Digest the cached files and the exact population the configured draw selects."""
    root = data_dir or DEFAULT_DATA_DIR
    relative = _DATASET_FILES.get(dataset)
    if relative is None:
        raise ValueError(f"No file list is known for dataset {dataset!r}")

    digests: list[FileDigest] = []
    for name in relative:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; download the dataset before locking it")
        digests.append(digest_file(path))

    everything = load(dataset, data_dir=data_dir)
    drawn = load(
        dataset, limit, data_dir=data_dir, split=split, seed=seed, stratify=stratify
    )
    ids = "\n".join(sample.id for sample in drawn).encode("utf-8")
    stats = describe(dataset, drawn)
    return DatasetLock(
        dataset=dataset,
        files=digests,
        total_samples=len(everything),
        sampling={"split": split, "seed": seed, "stratify": stratify, "limit": limit},
        drawn_samples=len(drawn),
        drawn_hallucinated=stats.hallucinated,
        drawn_sample_ids_sha256=hashlib.sha256(ids).hexdigest(),
        task_types=stats.task_types,
    )


def verify_lock(lock: DatasetLock, *, data_dir: Path | None = None) -> list[str]:
    """Differences between the lock and what is on this machine; empty means identical."""
    problems: list[str] = []
    root = data_dir or DEFAULT_DATA_DIR
    relative = _DATASET_FILES.get(lock.dataset, ())
    by_name = {digest.path: digest for digest in lock.files}

    for name in relative:
        path = root / name
        if not path.exists():
            problems.append(f"{path} is missing")
            continue
        expected = by_name.get(path.name)
        if expected is None:
            continue
        actual = digest_file(path)
        if actual.sha256 != expected.sha256:
            problems.append(
                f"{path.name}: sha256 {actual.sha256[:12]}… does not match the locked "
                f"{expected.sha256[:12]}… ({actual.bytes} bytes vs {expected.bytes})"
            )
    if problems:
        # The derived checks below would all fail too, and less informatively.
        return problems

    sampling = lock.sampling
    drawn = load(
        lock.dataset,
        sampling.get("limit"),
        data_dir=data_dir,
        split=sampling.get("split"),
        seed=sampling.get("seed"),
        stratify=bool(sampling.get("stratify")),
    )
    ids = hashlib.sha256("\n".join(sample.id for sample in drawn).encode("utf-8")).hexdigest()
    if len(drawn) != lock.drawn_samples:
        problems.append(f"draw size {len(drawn)} does not match the locked {lock.drawn_samples}")
    if ids != lock.drawn_sample_ids_sha256:
        problems.append("the drawn sample identities differ from the locked draw")
    hallucinated = sum(sample.label_hallucinated for sample in drawn)
    if hallucinated != lock.drawn_hallucinated:
        problems.append(
            f"label balance {hallucinated} positives does not match the locked "
            f"{lock.drawn_hallucinated}"
        )
    return problems


def read_lock(path: Path) -> DatasetLock:
    """Load a lock file written by `write`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"] = [FileDigest(**entry) for entry in payload["files"]]
    return DatasetLock(**payload)


@app.command()
def write(
    dataset: Annotated[str, typer.Option(help="Dataset to lock.")] = "ragtruth",
    out: Annotated[Path, typer.Option(help="Lock file to write.")] = DEFAULT_LOCK_PATH,
    split: Annotated[str | None, typer.Option(help="Split the experiment will use.")] = "test",
    limit: Annotated[int | None, typer.Option(help="Samples the experiment will draw.")] = 150,
    seed: Annotated[int, typer.Option(help="Sampling seed the experiment will use.")] = 20260806,
    stratify: Annotated[bool, typer.Option("--stratify/--no-stratify")] = True,
    data_dir: Annotated[Path | None, typer.Option(help="Dataset cache directory.")] = None,
) -> None:
    """Record digests of the cached dataset and of the exact draw the experiment will score."""
    lock = build_lock(
        dataset, data_dir=data_dir, limit=limit, split=split, seed=seed, stratify=stratify
    )
    out.write_text(json.dumps(asdict(lock), indent=2) + "\n", encoding="utf-8")
    typer.echo(f"locked {lock.dataset}: {lock.total_samples} samples total, "
               f"{lock.drawn_samples} drawn ({lock.drawn_hallucinated} hallucinated)")
    for digest in lock.files:
        typer.echo(f"  {digest.path}  {digest.sha256[:16]}…  {digest.bytes} bytes")
    typer.echo(f"wrote {out}")


@app.command()
def verify(
    lock_path: Annotated[Path, typer.Argument(help="Lock file to check against.")] = (
        DEFAULT_LOCK_PATH
    ),
    data_dir: Annotated[Path | None, typer.Option(help="Dataset cache directory.")] = None,
) -> None:
    """Fail unless this machine holds byte-identical data and draws the same samples."""
    if not lock_path.exists():
        typer.echo(f"{lock_path} does not exist; run `write` on the machine that has the data.")
        raise typer.Exit(code=1)

    problems = verify_lock(read_lock(lock_path), data_dir=data_dir)
    if problems:
        typer.echo("dataset does NOT match the lock:")
        for problem in problems:
            typer.echo(f"  - {problem}")
        raise typer.Exit(code=1)
    typer.echo("dataset matches the lock: same files, same draw, same label balance.")


if __name__ == "__main__":  # pragma: no cover
    app()
