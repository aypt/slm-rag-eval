"""Benchmark dataset loaders.

Both datasets are cached under `data/` (gitignored) by the scripts in `scripts/`; the loaders
themselves never touch the network, so a benchmark run is reproducible offline.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
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
    split: str | None = None,
    seed: int | None = None,
    stratify: bool = False,
) -> list[LabeledSample]:
    """Load a cached dataset by name, optionally filtered, sampled and truncated.

    `limit` alone takes the first N samples in file order, which is not a defensible test
    set: RAGTruth's file is ordered by source and response model, so the first 200 rows are
    biased towards whichever task type and generator happen to come first. Pass `split`,
    `seed` and `stratify` for a sample the report can defend.
    """
    loader = _LOADERS.get(name)
    if loader is None:
        available = ", ".join(sorted(_LOADERS))
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when given")

    samples = loader(data_dir or DEFAULT_DATA_DIR)
    # An empty `--split` means "no split filter". Datasets without official splits need a
    # way to say that from the command line, and treating "" as a split name would only
    # ever raise.
    if split:
        samples = select_split(samples, split)
    if seed is not None:
        samples = shuffled(samples, seed)
    if limit is None:
        return samples
    if stratify:
        return stratified_sample(samples, limit)
    return samples[:limit]


def available_splits(samples: Sequence[LabeledSample]) -> tuple[str, ...]:
    """Split names present in `samples`, for error messages and the dataset table."""
    return tuple(
        sorted({str(sample.meta["split"]) for sample in samples if sample.meta.get("split")})
    )


def select_split(samples: list[LabeledSample], split: str) -> list[LabeledSample]:
    """Keep only samples from one official split, failing loudly if it is not there.

    Silently returning everything when a split is misspelled would put training data into a
    table labelled "test", which is the kind of error a reader cannot detect afterwards.
    """
    selected = [sample for sample in samples if str(sample.meta.get("split")) == split]
    if not selected:
        present = available_splits(samples)
        hint = (
            f" Available splits: {', '.join(present)}."
            if present
            else " This dataset records no split."
        )
        raise ValueError(f"No samples in split {split!r}.{hint}")
    return selected


def shuffled(samples: list[LabeledSample], seed: int) -> list[LabeledSample]:
    """Deterministic shuffle. The seed goes in the manifest so the draw can be repeated."""
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    return ordered


def stratified_sample(samples: list[LabeledSample], limit: int) -> list[LabeledSample]:
    """Take `limit` samples keeping the label balance of the full set as closely as possible.

    An unstratified draw of 150 from a set that is 30% hallucinated can easily land at 20%
    or 40%, which moves every threshold metric for a reason that has nothing to do with the
    judge. Strata are (label, task_type) so RAGTruth's three task types stay represented.
    """
    if limit >= len(samples):
        return list(samples)

    strata: dict[tuple[bool, str], list[LabeledSample]] = {}
    for sample in samples:
        key = (sample.label_hallucinated, str(sample.meta.get("task_type", "")))
        strata.setdefault(key, []).append(sample)

    # Largest-remainder allocation: floor each quota, then hand the leftovers to the strata
    # that lost the most, so the totals add up to exactly `limit`.
    total = len(samples)
    exact = {key: len(group) * limit / total for key, group in strata.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    for key in sorted(strata, key=lambda k: (-(exact[k] - quotas[k]), k)):
        if remaining <= 0:
            break
        quotas[key] += 1
        remaining -= 1

    taken: list[LabeledSample] = []
    for key, group in strata.items():
        taken.extend(group[: quotas[key]])
    # Restore the incoming order so the result is a subsequence, not grouped by stratum.
    position = {id(sample): index for index, sample in enumerate(samples)}
    return sorted(taken, key=lambda sample: position[id(sample)])


@dataclass(frozen=True)
class DatasetStats:
    """The dataset table the report needs (Table 5.1), computed from what was actually loaded."""

    name: str
    total: int
    hallucinated: int
    splits: dict[str, int]
    task_types: dict[str, int]
    median_context_chars: int
    max_context_chars: int
    median_answer_chars: int

    @property
    def hallucinated_share(self) -> float:
        """Positive-class share; 0.0 for an empty set rather than a division error."""
        return self.hallucinated / self.total if self.total else 0.0


def describe(name: str, samples: Sequence[LabeledSample]) -> DatasetStats:
    """Summarize a loaded dataset for the methodology chapter."""
    context_lengths = [sum(len(text) for text in sample.contexts) for sample in samples] or [0]
    answer_lengths = [len(sample.answer) for sample in samples] or [0]
    splits: dict[str, int] = {}
    task_types: dict[str, int] = {}
    for sample in samples:
        split = str(sample.meta.get("split") or "unspecified")
        task = str(sample.meta.get("task_type") or "unspecified")
        splits[split] = splits.get(split, 0) + 1
        task_types[task] = task_types.get(task, 0) + 1
    return DatasetStats(
        name=name,
        total=len(samples),
        hallucinated=sum(sample.label_hallucinated for sample in samples),
        splits=dict(sorted(splits.items())),
        task_types=dict(sorted(task_types.items())),
        median_context_chars=int(statistics.median(context_lengths)),
        max_context_chars=max(context_lengths),
        median_answer_chars=int(statistics.median(answer_lengths)),
    )


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


app = typer.Typer(help="Inspect a cached dataset without running a judge.", add_completion=False)


@app.command()
def main(
    dataset: Annotated[str, typer.Option(help="Dataset to describe.")] = "ragtruth",
    split: Annotated[str | None, typer.Option(help="Restrict to one official split.")] = None,
    limit: Annotated[int | None, typer.Option(help="Describe only N samples.")] = None,
    seed: Annotated[int | None, typer.Option(help="Shuffle seed before limiting.")] = None,
    stratify: Annotated[
        bool, typer.Option("--stratify", help="Keep the label/task balance when limiting.")
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Dataset cache directory.")] = None,
) -> None:
    """Print the dataset statistics table (report Table 5.1) as Markdown."""
    samples = load(dataset, limit, data_dir=data_dir, split=split, seed=seed, stratify=stratify)
    stats = describe(dataset, samples)
    typer.echo(render_stats(stats))


def render_stats(stats: DatasetStats) -> str:
    """Markdown for the methodology chapter: sizes, balance, and context lengths."""
    rows = [
        ("Samples", str(stats.total)),
        (
            "Hallucinated / not",
            f"{stats.hallucinated} / {stats.total - stats.hallucinated} "
            f"({stats.hallucinated_share:.1%} positive)",
        ),
        ("Splits", ", ".join(f"{name}: {count}" for name, count in stats.splits.items())),
        (
            "Task types",
            ", ".join(f"{name}: {count}" for name, count in stats.task_types.items()),
        ),
        (
            "Context length (chars)",
            f"median {stats.median_context_chars}, max {stats.max_context_chars}",
        ),
        ("Answer length (chars)", f"median {stats.median_answer_chars}"),
    ]
    lines = [f"### {stats.name} dataset statistics", "", "| Property | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    app()
