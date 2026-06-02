"""Turn benchmark JSONL rows into detection-quality, agreement, and efficiency reports.

Input rows are what `bench.run` writes. A row whose `scores.faithfulness` is null is counted
as *unscored* and excluded from every threshold metric — silently treating it as a score would
invent data the judge never produced.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

matplotlib.use("Agg")  # Headless: figures are files, never windows.

THRESHOLD_STEP = 0.05
THRESHOLDS: tuple[float, ...] = tuple(round(i * THRESHOLD_STEP, 2) for i in range(21))

app = typer.Typer(help=__doc__, add_completion=False)


@dataclass(frozen=True)
class ThresholdMetrics:
    """Detection quality at one decision threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float


@dataclass
class JudgeReport:
    """Everything computed for a single reporting series."""

    judge: str
    model: str
    scored: int
    unscored: int
    positives: int
    negatives: int
    sweep: list[ThresholdMetrics]
    best: ThresholdMetrics | None
    roc_auc: float | None
    latency_median_ms: float | None
    latency_p95_ms: float | None
    mean_tokens: dict[str, float] = field(default_factory=dict)
    label: str = ""
    datasets: tuple[str, ...] = ()
    unlabeled: int = 0
    """Rows with a score but no human label; excluded from every quality metric."""
    failed: int = 0
    """Samples that produced no row at all, counted from the run manifests."""

    @property
    def series(self) -> str:
        """Name used in tables and figures; equals the judge unless it had to be qualified."""
        return self.label or self.judge


def manifest_path_for(rows_path: Path) -> Path:
    """The manifest `bench.run` writes beside a rows file."""
    return rows_path.with_name(f"{rows_path.stem}.manifest.json")


@dataclass(frozen=True)
class RunProvenance:
    """What the manifest beside a rows file says about how those rows were produced."""

    run_id: str | None
    failure_count: int
    failures: tuple[str, ...]
    git_sha: str | None
    git_dirty: bool | None

    @property
    def fingerprint(self) -> str:
        """Series qualifier: rows from different run ids are not comparable."""
        return self.run_id or ""


def load_provenance(rows_path: Path) -> RunProvenance:
    """Read the manifest beside `rows_path`; absent or unreadable means unknown provenance."""
    path = manifest_path_for(rows_path)
    if not path.exists():
        return RunProvenance(None, 0, (), None, None)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RunProvenance(None, 0, (), None, None)
    if not isinstance(manifest, dict):
        return RunProvenance(None, 0, (), None, None)

    raw_failures = manifest.get("failures")
    failures = tuple(
        str(item.get("sample_id"))
        for item in (raw_failures if isinstance(raw_failures, list) else [])
        if isinstance(item, dict)
    )
    count = manifest.get("failure_count")
    run_id = manifest.get("run_id")
    dirty = manifest.get("git_dirty")
    sha = manifest.get("git_sha")
    return RunProvenance(
        run_id=run_id if isinstance(run_id, str) else None,
        failure_count=count if isinstance(count, int) else len(failures),
        failures=failures,
        git_sha=sha if isinstance(sha, str) else None,
        git_dirty=dirty if isinstance(dirty, bool) else None,
    )


def load_rows(paths: Iterable[Path]) -> pd.DataFrame:
    """Read benchmark JSONL files into one flat frame, carrying each file's provenance.

    A row with no `label_hallucinated` key is *unlabeled*, not negative. Coercing a missing
    label with `bool()` turned every unlabeled row — every row `rageval batch` writes — into
    a human-verified non-hallucination, which then scored as a true negative.
    """
    records: list[dict[str, Any]] = []
    for path in paths:
        rows_path = Path(path)
        provenance = load_provenance(rows_path)
        with rows_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                scores = row.get("scores") or {}
                usage = row.get("usage") or {}
                raw_label = row.get("label_hallucinated")
                records.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "dataset": row.get("dataset"),
                        "judge": row.get("judge"),
                        "model": row.get("model"),
                        "label_hallucinated": None if raw_label is None else bool(raw_label),
                        "faithfulness": scores.get("faithfulness"),
                        "relevance": scores.get("relevance"),
                        "latency_ms": row.get("latency_ms"),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "run_id": provenance.run_id,
                        "run_failures": provenance.failure_count,
                        "source_path": str(rows_path),
                    }
                )
    if not records:
        raise ValueError("No benchmark rows found in the given files")
    return pd.DataFrame.from_records(records)


def threshold_metrics(
    labels: Sequence[bool],
    faithfulness: Sequence[float],
    threshold: float,
) -> ThresholdMetrics:
    """Confusion counts and rates for `predicted_hallucinated := faithfulness < threshold`."""
    predicted = [score < threshold for score in faithfulness]
    true_positives = sum(1 for p, y in zip(predicted, labels, strict=True) if p and y)
    false_positives = sum(1 for p, y in zip(predicted, labels, strict=True) if p and not y)
    false_negatives = sum(1 for p, y in zip(predicted, labels, strict=True) if not p and y)
    true_negatives = sum(1 for p, y in zip(predicted, labels, strict=True) if not p and not y)

    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    specificity = _ratio(true_negatives, true_negatives + false_positives)
    return ThresholdMetrics(
        threshold=threshold,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=(recall + specificity) / 2,
    )


def _ratio(numerator: int, denominator: int) -> float:
    """0.0 for an empty denominator: no predictions means no measured quality."""
    return numerator / denominator if denominator else 0.0


SERIES_COLUMN = "series"


def assign_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the reporting-series column: the judge, qualified when it spans models/datasets.

    Grouping on `judge` alone silently pools rows from different models or different
    datasets into one set of metrics and labels them with whichever model appeared first.
    A judge that stays on one model and one dataset keeps its plain name, so the common
    case reads exactly as before.
    """
    aggregations: dict[str, tuple[str, str]] = {
        "models": ("model", "nunique"),
        "datasets": ("dataset", "nunique"),
    }
    has_run_id = "run_id" in frame.columns
    if has_run_id:
        aggregations["runs"] = ("run_id", "nunique")
    spans = frame.groupby("judge").agg(**aggregations)

    def label(row: pd.Series) -> str:
        judge = str(row["judge"])
        parts = [judge]
        if spans.loc[judge, "models"] > 1:
            parts.append(f":{row['model']}")
        if spans.loc[judge, "datasets"] > 1:
            parts.append(f"@{row['dataset']}")
        # Two runs of the same judge, model and dataset still differ if anything in the run
        # fingerprint differs — privacy mode above all. Pooling a masked run with an
        # unmasked one would average the ablation away instead of measuring it.
        if has_run_id and spans.loc[judge, "runs"] > 1 and row["run_id"]:
            parts.append(f"#{row['run_id']}")
        return "".join(parts)

    labelled = frame.copy()
    labelled[SERIES_COLUMN] = labelled.apply(label, axis=1)
    return labelled


def _series_column(frame: pd.DataFrame) -> str:
    """Fall back to `judge` for frames that were not passed through `assign_series`."""
    return SERIES_COLUMN if SERIES_COLUMN in frame.columns else "judge"


def _failed_samples(frame: pd.DataFrame) -> int:
    """Samples the judge could not score at all, per the manifests behind these rows.

    A failed sample writes no row, so it is invisible in the JSONL. Counting it here keeps
    the denominator honest: a judge that fails on the samples it finds hardest would
    otherwise post better numbers precisely because the hard cases went missing.
    """
    if "run_failures" not in frame.columns or "source_path" not in frame.columns:
        return 0
    per_file = frame.drop_duplicates(subset="source_path")["run_failures"]
    return int(per_file.fillna(0).sum())


def _labeled_scored(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry both a judge score and a human label — the only ones quality uses."""
    return frame[frame["faithfulness"].notna() & frame["label_hallucinated"].notna()]


def analyze_judge(frame: pd.DataFrame) -> JudgeReport:
    """Sweep thresholds, pick the best-F1 one, and summarize efficiency for one judge."""
    scored = frame[frame["faithfulness"].notna()]
    # Detection quality needs a ground truth. Rows with a score but no label are reported
    # separately instead of being folded in as negatives.
    labeled = _labeled_scored(frame)
    labels = [bool(value) for value in labeled["label_hallucinated"]]
    scores = [float(value) for value in labeled["faithfulness"]]

    sweep = [threshold_metrics(labels, scores, threshold) for threshold in THRESHOLDS]
    # Ties go to the lower threshold: fewer positives for the same F1.
    best = max(sweep, key=lambda m: (m.f1, -m.threshold)) if scores else None

    roc_auc: float | None = None
    if len(set(labels)) == 2:
        # Lower faithfulness means "more likely hallucinated", so the score is negated.
        roc_auc = float(roc_auc_score(labels, [-score for score in scores]))

    latencies = frame["latency_ms"].dropna().to_numpy(dtype=float)
    models = sorted({str(value) for value in frame["model"].dropna().unique()})
    return JudgeReport(
        judge=str(frame["judge"].iloc[0]),
        # Every row in a series shares one model; joining is a visible signal if it ever does not.
        model=" + ".join(models) if models else "unknown",
        label=str(frame[_series_column(frame)].iloc[0]),
        datasets=tuple(sorted({str(value) for value in frame["dataset"].dropna().unique()})),
        scored=len(scored),
        unscored=int(frame["faithfulness"].isna().sum()),
        unlabeled=int(frame["label_hallucinated"].isna().sum()),
        failed=_failed_samples(frame),
        positives=sum(labels),
        negatives=len(labels) - sum(labels),
        sweep=sweep,
        best=best,
        roc_auc=roc_auc,
        latency_median_ms=float(np.median(latencies)) if latencies.size else None,
        latency_p95_ms=float(np.percentile(latencies, 95)) if latencies.size else None,
        mean_tokens={
            column: float(frame[column].mean())
            for column in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    )


@dataclass(frozen=True)
class AgreementReport:
    """SLM-vs-cloud agreement over the samples both judges scored."""

    judge_a: str
    judge_b: str
    overlap: int
    pearson: float | None
    spearman: float | None
    cohen_kappa: float | None


def analyze_agreement(
    frame: pd.DataFrame,
    reports: dict[str, JudgeReport],
) -> list[AgreementReport]:
    """Correlate faithfulness and compare binary calls for every pair of judges."""
    agreements: list[AgreementReport] = []
    judges = sorted(reports)
    for index, judge_a in enumerate(judges):
        for judge_b in judges[index + 1 :]:
            left = _scored_by_sample(frame, judge_a)
            right = _scored_by_sample(frame, judge_b)
            shared = sorted(set(left) & set(right))
            if not shared:
                continue

            scores_a = [left[sample_id] for sample_id in shared]
            scores_b = [right[sample_id] for sample_id in shared]
            best_a = reports[judge_a].best
            best_b = reports[judge_b].best
            kappa: float | None = None
            if best_a is not None and best_b is not None:
                calls_a = [score < best_a.threshold for score in scores_a]
                calls_b = [score < best_b.threshold for score in scores_b]
                if len(set(calls_a)) > 1 or len(set(calls_b)) > 1:
                    kappa = float(cohen_kappa_score(calls_a, calls_b))
                # When neither judge varies, kappa is undefined: chance agreement is 1, so
                # the correction divides by zero. It is reported as n/a rather than as
                # perfect agreement, which is what a constant-vs-constant 1.0 would claim.

            agreements.append(
                AgreementReport(
                    judge_a=judge_a,
                    judge_b=judge_b,
                    overlap=len(shared),
                    pearson=pearson(scores_a, scores_b),
                    spearman=spearman(scores_a, scores_b),
                    cohen_kappa=kappa,
                )
            )
    return agreements


def _scored_by_sample(frame: pd.DataFrame, series: str) -> dict[tuple[str, str], float]:
    """Scores keyed by (dataset, sample_id): the same id in two datasets is two samples."""
    column = _series_column(frame)
    subset = frame[(frame[column] == series) & (frame["faithfulness"].notna())]
    return {
        (str(dataset), str(sample_id)): float(score)
        for dataset, sample_id, score in zip(
            subset["dataset"], subset["sample_id"], subset["faithfulness"], strict=True
        )
    }


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation, or None when it is undefined (< 2 points, or a constant series)."""
    if len(left) < 2 or len(set(left)) == 1 or len(set(right)) == 1:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return None if np.isnan(value) else value


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman correlation: Pearson on average-tied ranks."""
    if len(left) < 2 or len(set(left)) == 1 or len(set(right)) == 1:
        return None
    ranked_left = pd.Series(left).rank().to_list()
    ranked_right = pd.Series(right).rank().to_list()
    value = float(np.corrcoef(ranked_left, ranked_right)[0, 1])
    return None if np.isnan(value) else value


@dataclass(frozen=True)
class CostModel:
    """Cloud pricing in USD per one million tokens; the local judge is marginal-cost zero."""

    cloud_input_per_1m: float = 0.0
    cloud_output_per_1m: float = 0.0

    def estimate(self, report: JudgeReport, samples: int) -> float:
        """Estimated USD for `samples` requests at this judge's mean token usage."""
        if report.judge == "slm":
            return 0.0
        prompt = report.mean_tokens.get("prompt_tokens", 0.0) * samples
        completion = report.mean_tokens.get("completion_tokens", 0.0) * samples
        return (
            prompt * self.cloud_input_per_1m + completion * self.cloud_output_per_1m
        ) / 1_000_000


def _figure_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.png"


def plot_roc_curves(frame: pd.DataFrame, reports: dict[str, JudgeReport], out_dir: Path) -> Path:
    """One figure, one line per judge, plus the chance diagonal for reference."""
    figure, axes = plt.subplots(figsize=(6, 5))
    # Line style varies with the judge so the curves stay distinguishable without color.
    styles = ("-", "--", "-.", ":")
    plotted = False
    for index, (judge, report) in enumerate(sorted(reports.items())):
        column = _series_column(frame)
        scored = _labeled_scored(frame[frame[column] == judge])
        labels = [bool(value) for value in scored["label_hallucinated"]]
        if len(set(labels)) != 2:
            continue
        scores = [-float(value) for value in scored["faithfulness"]]
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        auc_text = "n/a" if report.roc_auc is None else f"{report.roc_auc:.3f}"
        axes.plot(
            false_positive_rate,
            true_positive_rate,
            styles[index % len(styles)],
            linewidth=2,
            label=f"{judge} (AUC {auc_text})",
        )
        plotted = True

    axes.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":", label="chance")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title("Hallucination detection ROC")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.grid(alpha=0.3)
    if plotted:
        axes.legend(loc="lower right")
    return _save(figure, _figure_path(out_dir, "roc_curves"))


def plot_score_distributions(frame: pd.DataFrame, out_dir: Path) -> Path:
    """Faithfulness histograms, one panel per judge, on a shared 0–1 axis."""
    column = _series_column(frame)
    judges = sorted(frame[column].dropna().unique())
    figure, axes_list = plt.subplots(
        len(judges), 1, figsize=(6, 3 * max(len(judges), 1)), squeeze=False, sharex=True
    )
    bins = np.linspace(0, 1, 11)
    for axes, judge in zip(axes_list[:, 0], judges, strict=True):
        scored = frame[(frame[column] == judge) & (frame["faithfulness"].notna())]
        axes.hist(scored["faithfulness"].to_numpy(dtype=float), bins=bins)
        axes.set_title(f"{judge} faithfulness")
        axes.set_ylabel("Samples")
        axes.grid(alpha=0.3)
    axes_list[-1, 0].set_xlabel("Faithfulness score")
    return _save(figure, _figure_path(out_dir, "score_distributions"))


def plot_latency_box(frame: pd.DataFrame, out_dir: Path) -> Path:
    """Per-sample latency spread per judge."""
    column = _series_column(frame)
    judges = sorted(frame[column].dropna().unique())
    series = [
        frame[frame[column] == judge]["latency_ms"].dropna().to_numpy(dtype=float)
        for judge in judges
    ]
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.boxplot([values for values in series if values.size], tick_labels=judges)
    axes.set_ylabel("Latency per sample (ms)")
    axes.set_title("Judge latency")
    axes.grid(alpha=0.3, axis="y")
    return _save(figure, _figure_path(out_dir, "latency_box"))


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _format(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_summary(
    frame: pd.DataFrame,
    reports: dict[str, JudgeReport],
    agreements: Sequence[AgreementReport],
    costs: CostModel,
    figures: Sequence[Path],
) -> str:
    """Build report/summary.md: detection quality, agreement, efficiency, and cost."""
    datasets = ", ".join(sorted({str(value) for value in frame["dataset"].dropna().unique()}))
    sections = [
        "# Benchmark summary",
        "",
        f"Datasets: {datasets or 'n/a'} · rows: {len(frame)} · judges: "
        f"{', '.join(sorted(reports))}",
        "",
        "Predicted hallucinated when `faithfulness < threshold`. Rows the judge could not "
        "score are reported as *unscored* and excluded from every threshold metric.",
        "",
        "*Coverage* columns are part of the result, not bookkeeping. **Unscored** rows "
        "produced no score, **Unlabeled** rows carry no human label (both are excluded from "
        "every quality metric), and **Failed** samples produced no row at all — they come "
        "from the run manifests, because a sample that is missing from the rows file would "
        "otherwise shrink the denominator invisibly. Quote P/R/F1 together with these "
        "counts or the numbers describe an unstated subset.",
        "",
        "## Detection quality at the best-F1 threshold",
        "",
        _markdown_table(
            [
                "Judge",
                "Model",
                "Scored",
                "Unscored",
                "Unlabeled",
                "Failed",
                "Best t",
                "P",
                "R",
                "F1",
                "BA",
                "ROC-AUC",
            ],
            [
                [
                    report.series,
                    report.model,
                    str(report.scored),
                    str(report.unscored),
                    str(report.unlabeled),
                    str(report.failed),
                    _format(report.best.threshold, 2) if report.best else "n/a",
                    _format(report.best.precision) if report.best else "n/a",
                    _format(report.best.recall) if report.best else "n/a",
                    _format(report.best.f1) if report.best else "n/a",
                    _format(report.best.balanced_accuracy) if report.best else "n/a",
                    _format(report.roc_auc),
                ]
                for report in reports.values()
            ],
        ),
        "",
    ]

    for report in reports.values():
        sections += [
            f"### {report.series}: threshold sweep",
            "",
            _markdown_table(
                ["t", "TP", "FP", "FN", "TN", "P", "R", "F1", "BA"],
                [
                    [
                        _format(metrics.threshold, 2),
                        str(metrics.true_positives),
                        str(metrics.false_positives),
                        str(metrics.false_negatives),
                        str(metrics.true_negatives),
                        _format(metrics.precision),
                        _format(metrics.recall),
                        _format(metrics.f1),
                        _format(metrics.balanced_accuracy),
                    ]
                    for metrics in report.sweep
                ],
            ),
            "",
        ]

    sections += ["## Judge agreement", ""]
    if agreements:
        sections += [
            _markdown_table(
                ["Pair", "Shared samples", "Pearson", "Spearman", "Cohen's kappa"],
                [
                    [
                        f"{item.judge_a} vs {item.judge_b}",
                        str(item.overlap),
                        _format(item.pearson),
                        _format(item.spearman),
                        _format(item.cohen_kappa),
                    ]
                    for item in agreements
                ],
            ),
            "",
            "Kappa compares the binary calls each judge makes at its **own** best-F1 threshold.",
            "",
        ]
    else:
        sections += ["No sample was scored by more than one judge.", ""]

    sections += [
        "## Efficiency and cost",
        "",
        _markdown_table(
            ["Judge", "Median latency (ms)", "p95 latency (ms)", "Mean tokens", "Est. cost (USD)"],
            [
                [
                    report.series,
                    _format(report.latency_median_ms, 1),
                    _format(report.latency_p95_ms, 1),
                    _format(report.mean_tokens.get("total_tokens"), 1),
                    _format(costs.estimate(report, report.scored + report.unscored), 4),
                ]
                for report in reports.values()
            ],
        ),
        "",
        "Latency percentiles use linear interpolation between the two nearest samples. "
        "The local judge is reported at zero marginal cost: it bills no tokens, but it does "
        "occupy hardware you already pay for, so its true cost is amortized GPU/CPU time.",
        "",
        "## Figures",
        "",
    ]
    sections += [f"![{path.stem}]({path.name})" for path in figures]
    sections.append("")
    return "\n".join(sections)


def analyze(
    paths: Sequence[Path],
    out_dir: Path,
    *,
    costs: CostModel | None = None,
) -> dict[str, Any]:
    """Full pipeline: load rows, compute every metric, write figures and summary.md."""
    frame = load_rows(paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_model = costs or CostModel()

    frame = assign_series(frame)
    reports = {
        str(series): analyze_judge(frame[frame[SERIES_COLUMN] == series])
        for series in sorted(frame[SERIES_COLUMN].dropna().unique())
    }
    agreements = analyze_agreement(frame, reports)
    figures = [
        plot_roc_curves(frame, reports, out_dir),
        plot_score_distributions(frame, out_dir),
        plot_latency_box(frame, out_dir),
    ]

    summary_path = out_dir / "summary.md"
    summary_path.write_text(
        render_summary(frame, reports, agreements, cost_model, figures), encoding="utf-8"
    )
    return {
        "summary_path": summary_path,
        "figures": figures,
        "reports": reports,
        "agreements": agreements,
    }


@app.command()
def main(
    results: Annotated[list[Path], typer.Argument(help="Benchmark JSONL files to analyze.")],
    out: Annotated[Path, typer.Option(help="Directory for summary.md and figures.")] = Path(
        "report"
    ),
    cloud_input_cost_per_1m: Annotated[
        float, typer.Option(help="Cloud judge USD per 1M input tokens.")
    ] = 0.0,
    cloud_output_cost_per_1m: Annotated[
        float, typer.Option(help="Cloud judge USD per 1M output tokens.")
    ] = 0.0,
) -> None:
    """Analyze benchmark rows and write report/summary.md plus PNG figures."""
    outcome = analyze(
        results,
        out,
        costs=CostModel(
            cloud_input_per_1m=cloud_input_cost_per_1m,
            cloud_output_per_1m=cloud_output_cost_per_1m,
        ),
    )
    typer.echo(str(outcome["summary_path"]))
    for figure in outcome["figures"]:
        typer.echo(str(figure))


if __name__ == "__main__":  # pragma: no cover
    app()
