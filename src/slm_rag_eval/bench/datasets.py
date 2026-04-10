"""Benchmark dataset loaders.

Both datasets are cached under `data/` (gitignored) by the scripts in `scripts/`; the loaders
themselves never touch the network, so a benchmark run is reproducible offline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_DATA_DIR = Path("data")

_PASSAGE_SEPARATOR = re.compile(r"passage\s+\d+\s*:", re.IGNORECASE)


class LabeledSample(BaseModel):
    """One human-labeled RAG interaction, normalized across datasets."""

    id: str
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    label_hallucinated: bool
    meta: dict[str, Any] = Field(default_factory=dict)


class DatasetNotDownloadedError(FileNotFoundError):
    """Raised when a dataset's cached files are missing, naming the script that fetches them."""

    def __init__(self, dataset: str, missing: Path, script: str) -> None:
        super().__init__(
            f"{dataset} is not downloaded: {missing} is missing. "
            f"Run `python {script}` first (it writes into {missing.parent})."
        )
        self.dataset = dataset
        self.missing = missing
        self.script = script


def load(
    name: str,
    limit: int | None = None,
    *,
    data_dir: Path | None = None,
) -> list[LabeledSample]:
    """Load a cached dataset by name, optionally truncated to the first `limit` samples."""
    loader = _LOADERS.get(name)
    if loader is None:
        available = ", ".join(sorted(_LOADERS))
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when given")

    samples = loader(data_dir or DEFAULT_DATA_DIR)
    return samples[:limit] if limit is not None else samples


def available_datasets() -> tuple[str, ...]:
    """Names accepted by `load`."""
    return tuple(sorted(_LOADERS))


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    yield parsed


def _split_passages(passages: str) -> list[str]:
    """Split RAGTruth's `passage 1:… passage 2:…` blob; a blob without markers stays whole."""
    parts = [part.strip() for part in _PASSAGE_SEPARATOR.split(passages)]
    non_empty = [part for part in parts if part]
    return non_empty or [passages.strip()]


def _ragtruth_question_and_contexts(source: dict[str, Any]) -> tuple[str, list[str]]:
    """Normalize the three RAGTruth task types onto one question/contexts shape."""
    info = source.get("source_info")
    prompt = str(source.get("prompt", "")).strip()

    if isinstance(info, dict):
        question = str(info.get("question", "")).strip() or prompt
        passages = info.get("passages")
        if isinstance(passages, str):
            return question, _split_passages(passages)
        # Structured records (data-to-text): keep the whole record as one context.
        return question, [json.dumps(info, ensure_ascii=False, sort_keys=True)]

    # Summarization: the source document is the only context and the prompt is the task.
    document = str(info).strip() if info is not None else ""
    return prompt, [document] if document else []


def _load_ragtruth(data_dir: Path) -> list[LabeledSample]:
    root = data_dir / "ragtruth"
    responses_path = root / "response.jsonl"
    sources_path = root / "source_info.jsonl"
    for path in (sources_path, responses_path):
        if not path.exists():
            raise DatasetNotDownloadedError("ragtruth", path, "scripts/download_ragtruth.py")

    sources = {str(row["source_id"]): row for row in _read_jsonl(sources_path)}

    samples: list[LabeledSample] = []
    for response in _read_jsonl(responses_path):
        source = sources.get(str(response.get("source_id")))
        if source is None:
            continue
        question, contexts = _ragtruth_question_and_contexts(source)
        # Response-level label: RAGTruth annotates spans, so any span means hallucinated.
        labels = response.get("labels") or []
        samples.append(
            LabeledSample(
                id=str(response["id"]),
                question=question,
                answer=str(response.get("response", "")),
                contexts=contexts,
                label_hallucinated=bool(labels),
                meta={
                    "dataset": "ragtruth",
                    "source_id": str(response.get("source_id")),
                    "task_type": source.get("task_type"),
                    "source": source.get("source"),
                    "response_model": response.get("model"),
                    "split": response.get("split"),
                    "quality": response.get("quality"),
                    "label_count": len(labels),
                },
            )
        )
    return samples


def _load_halueval(data_dir: Path) -> list[LabeledSample]:
    path = data_dir / "halueval" / "qa_data.json"
    if not path.exists():
        raise DatasetNotDownloadedError("halueval", path, "scripts/download_halueval.py")

    samples: list[LabeledSample] = []
    for index, row in enumerate(_read_jsonl(path)):
        question = str(row.get("question", ""))
        knowledge = str(row.get("knowledge", ""))
        # Each source row carries a correct and a hallucinated answer for the same question,
        # which is exactly one negative and one positive sample.
        for suffix, key, hallucinated in (
            ("right", "right_answer", False),
            ("hallucinated", "hallucinated_answer", True),
        ):
            answer = row.get(key)
            if answer is None:
                continue
            samples.append(
                LabeledSample(
                    id=f"{index}-{suffix}",
                    question=question,
                    answer=str(answer),
                    contexts=[knowledge] if knowledge else [],
                    label_hallucinated=hallucinated,
                    meta={"dataset": "halueval", "subset": "qa", "row_index": index},
                )
            )
    return samples


_LOADERS: dict[str, Callable[[Path], list[LabeledSample]]] = {
    "ragtruth": _load_ragtruth,
    "halueval": _load_halueval,
}
