"""Fetch the benchmark datasets into the local cache.

Kept in the package (rather than only in `scripts/`) so the logic is importable and testable;
`scripts/download_*.py` are thin entry points over these functions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from slm_rag_eval.bench.datasets import DEFAULT_DATA_DIR

RAGTRUTH_BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
HALUEVAL_BASE = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data"

DATASET_SOURCES: dict[str, dict[str, str]] = {
    "ragtruth": {
        "response.jsonl": f"{RAGTRUTH_BASE}/response.jsonl",
        "source_info.jsonl": f"{RAGTRUTH_BASE}/source_info.jsonl",
    },
    "halueval": {
        "qa_data.json": f"{HALUEVAL_BASE}/qa_data.json",
    },
}

Fetcher = Callable[[str, Path], None]


def stream_to_file(url: str, destination: Path) -> None:
    """Download one file, writing to a temporary name so a partial file is never cached."""
    temporary = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    temporary.replace(destination)


def download_dataset(
    name: str,
    *,
    data_dir: Path | None = None,
    force: bool = False,
    fetch: Fetcher = stream_to_file,
) -> list[Path]:
    """Download every file of `name` that is not cached yet; return all cached paths."""
    sources = DATASET_SOURCES.get(name)
    if sources is None:
        available = ", ".join(sorted(DATASET_SOURCES))
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")

    target_dir = (data_dir or DEFAULT_DATA_DIR) / name
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for filename, url in sources.items():
        destination = target_dir / filename
        if destination.exists() and not force:
            print(f"cached: {destination}")
        else:
            print(f"downloading {url} -> {destination}")
            fetch(url, destination)
        written.append(destination)
    return written
