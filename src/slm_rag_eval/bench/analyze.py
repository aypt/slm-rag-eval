"""Turn benchmark JSONL rows into detection-quality, agreement, and efficiency reports.

Input rows are what `bench.run` writes. A row whose `scores.faithfulness` is null is counted
as *unscored* and excluded from every threshold metric — silently treating it as a score would
invent data the judge never produced.
"""

from __future__ import annotations

import hashlib
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
    accuracy: float = 0.0
    cohen_kappa: float | None = None
    """Agreement with the gold labels, corrected for chance; None when it is undefined."""

    @property
    def total(self) -> int:
        """Samples behind these counts — the denominator every rate here shares."""
        return (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )


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
    """Samples the judge could not score, counted from the error each failed row carries."""
    holdout: HoldoutResult | None = None
    """Threshold chosen on half the labeled rows and measured on the other half."""

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
    privacy_mode: str | None = None
    """`mask` or `off`; the axis the privacy ablation compares along."""

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
    identity = manifest.get("run_identity")
    privacy_mode = identity.get("privacy_mode") if isinstance(identity, dict) else None
    return RunProvenance(
        run_id=run_id if isinstance(run_id, str) else None,
        failure_count=count if isinstance(count, int) else len(failures),
        failures=failures,
        git_sha=sha if isinstance(sha, str) else None,
        git_dirty=dirty if isinstance(dirty, bool) else None,
        privacy_mode=privacy_mode if isinstance(privacy_mode, str) else None,
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
                        "error": row.get("error"),
                        "run_id": provenance.run_id,
                        "privacy_mode": provenance.privacy_mode,
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
    total = len(predicted)
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
        accuracy=_ratio(true_positives + true_negatives, total),
        cohen_kappa=_kappa_against_gold(labels, predicted),
    )


def _kappa_against_gold(labels: Sequence[bool], predicted: Sequence[bool]) -> float | None:
    """Chance-corrected agreement with the human labels, or None where it is undefined.

    Raw accuracy flatters a judge on an unbalanced set: calling everything faithful scores
    70% when 30% of samples are hallucinated. Kappa removes exactly that, which is why the
    rubric asks for it. It needs both vectors to vary; when neither does, it is undefined
    and stays None rather than being invented.
    """
    if len(set(labels)) < 2 and len(set(predicted)) < 2:
        return None
    value = float(cohen_kappa_score(list(labels), list(predicted)))
    return None if np.isnan(value) else value


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
    """Samples the judge could not score, counted from the error each failed row carries.

    Keeping the denominator honest matters more here than anywhere else: a judge that fails
    on the samples it finds hardest would post better numbers precisely because the hard
    cases went missing.
    """
    if "error" not in frame.columns:
        return 0
    return int(frame["error"].notna().sum())


def _labeled_scored(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry both a judge score and a human label — the only ones quality uses."""
    return frame[frame["faithfulness"].notna() & frame["label_hallucinated"].notna()]


HOLDOUT_SALT = "slm-rag-eval/threshold-split/v1"


def holdout_fold(dataset: str, sample_id: str) -> int:
    """Which half a sample belongs to: 0 selects the threshold, 1 reports the result.

    Hashed from the sample identity rather than drawn at random, for two reasons. Every
    judge must be scored on the *same* reporting half, or the comparison between them is
    not like for like. And the split has to survive a resume, a re-run, and a re-analysis
    without being stored anywhere.
    """
    digest = hashlib.sha256(f"{HOLDOUT_SALT}|{dataset}|{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 2


def _fold_masks(frame: pd.DataFrame) -> tuple[list[bool], list[bool]]:
    folds = [
        holdout_fold(str(dataset), str(sample_id))
        for dataset, sample_id in zip(frame["dataset"], frame["sample_id"], strict=True)
    ]
    return [fold == 0 for fold in folds], [fold == 1 for fold in folds]


@dataclass(frozen=True)
class HoldoutResult:
    """A threshold chosen on one half of the labeled data and measured on the other.

    The best-F1 threshold reported on the same rows it was chosen from is an upper bound,
    not a deployment setting — it has seen every sample it is scored on. This is the honest
    number: chosen blind to the rows it is measured on.
    """

    threshold: float
    selection_size: int
    reporting_size: int
    metrics: ThresholdMetrics


def select_threshold_holdout(
    labels: Sequence[bool],
    scores: Sequence[float],
    selection: Sequence[bool],
) -> HoldoutResult | None:
    """Pick best-F1 on the selection half, then measure it on the reporting half."""
    fit_labels = [label for label, chosen in zip(labels, selection, strict=True) if chosen]
    fit_scores = [score for score, chosen in zip(scores, selection, strict=True) if chosen]
    test_labels = [label for label, chosen in zip(labels, selection, strict=True) if not chosen]
    test_scores = [score for score, chosen in zip(scores, selection, strict=True) if not chosen]
    # Both halves need both classes. Without a positive in the selection half every
    # threshold scores F1 0 and the sweep silently returns 0.00, which reads in the table
    # as "this judge detects nothing" when the truth is "this split cannot measure it".
    if len(set(fit_labels)) < 2 or len(set(test_labels)) < 2:
        return None
    if not fit_scores or not test_scores:
        return None

    sweep = [threshold_metrics(fit_labels, fit_scores, threshold) for threshold in THRESHOLDS]
    chosen = max(sweep, key=lambda m: (m.f1, -m.threshold))
    return HoldoutResult(
        threshold=chosen.threshold,
        selection_size=len(fit_scores),
        reporting_size=len(test_scores),
        metrics=threshold_metrics(test_labels, test_scores, chosen.threshold),
    )


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
    selection_mask, _ = _fold_masks(labeled)
    holdout = select_threshold_holdout(labels, scores, selection_mask) if scores else None

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
        holdout=holdout,
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
class PrivacyAblation:
    """What masking cost this judge: the same samples scored with and without it.

    This is the answer to "is privacy protection free?", and it is the only number in the
    report that can justify running the pipeline in `mask` mode by default. A delta near
    zero says the judge reasons about `<PERSON_1>` as well as about a name; a large negative
    delta says the privacy layer is buying protection with accuracy.
    """

    judge: str
    model: str
    masked_series: str
    unmasked_series: str
    shared_samples: int
    masked_f1: float | None
    unmasked_f1: float | None
    masked_accuracy: float | None
    unmasked_accuracy: float | None

    @property
    def delta_f1(self) -> float | None:
        """Masked minus unmasked. Negative means masking cost accuracy."""
        if self.masked_f1 is None or self.unmasked_f1 is None:
            return None
        return self.masked_f1 - self.unmasked_f1

    @property
    def delta_accuracy(self) -> float | None:
        """Masked minus unmasked accuracy."""
        if self.masked_accuracy is None or self.unmasked_accuracy is None:
            return None
        return self.masked_accuracy - self.unmasked_accuracy


def analyze_privacy_ablation(
    frame: pd.DataFrame,
    reports: dict[str, JudgeReport],
) -> list[PrivacyAblation]:
    """Pair every masked series with the unmasked series of the same judge, model and dataset.

    Pairing is on (judge, model, dataset) rather than on the series label, because the two
    runs deliberately carry different run fingerprints — that is exactly what keeps the
    analyzer from pooling them in the first place.
    """
    if "privacy_mode" not in frame.columns:
        return []

    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    for series, report in reports.items():
        rows = frame[frame[_series_column(frame)] == series]
        modes = {str(value) for value in rows["privacy_mode"].dropna().unique()}
        if len(modes) != 1:
            continue
        key = (report.judge, report.model, ",".join(report.datasets))
        keyed.setdefault(key, {})[modes.pop()] = series

    ablations: list[PrivacyAblation] = []
    for (judge, model, _), by_mode in sorted(keyed.items()):
        masked, unmasked = by_mode.get("mask"), by_mode.get("off")
        if masked is None or unmasked is None:
            continue
        shared = set(_scored_by_sample(frame, masked)) & set(_scored_by_sample(frame, unmasked))
        ablations.append(
            PrivacyAblation(
                judge=judge,
                model=model,
                masked_series=masked,
                unmasked_series=unmasked,
                shared_samples=len(shared),
                masked_f1=_holdout_or_best(reports[masked], "f1"),
                unmasked_f1=_holdout_or_best(reports[unmasked], "f1"),
                masked_accuracy=_holdout_or_best(reports[masked], "accuracy"),
                unmasked_accuracy=_holdout_or_best(reports[unmasked], "accuracy"),
            )
        )
    return ablations


def _holdout_or_best(report: JudgeReport, attribute: str) -> float | None:
    """Held-out metric where one exists, otherwise the in-sample one; None if neither does."""
    metrics = report.holdout.metrics if report.holdout is not None else report.best
    if metrics is None:
        return None
    value = getattr(metrics, attribute)
    return float(value) if value is not None else None


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


def _best_f1(report: JudgeReport) -> float:
    """In-sample F1, or 0.0 when the series has no labeled rows to have measured one."""
    return report.best.f1 if report.best is not None else 0.0


def _holdout_f1(report: JudgeReport) -> float:
    """Held-out F1, or 0.0 when the labeled rows could not be split into two halves."""
    return report.holdout.metrics.f1 if report.holdout is not None else 0.0


def plot_reliability(reports: dict[str, JudgeReport], out_dir: Path) -> Path:
    """Report Figure 6.1: detection quality per judge, in-sample beside held-out.

    Showing both is the point. The gap between them is how much the best-F1 threshold owes
    to having seen the rows it is scored on, and a reader can only judge that if both are
    on the same axes.
    """
    series = sorted(reports)
    figure, axes = plt.subplots(figsize=(max(6, 1.6 * len(series) + 3), 4.5))
    positions = np.arange(len(series))
    width = 0.35

    in_sample = [_best_f1(reports[name]) for name in series]
    held_out = [_holdout_f1(reports[name]) for name in series]
    axes.bar(positions - width / 2, in_sample, width, label="F1 (in-sample threshold)",
             color="#4C72B0", edgecolor="black", linewidth=0.5)
    axes.bar(positions + width / 2, held_out, width, label="F1 (held-out threshold)",
             color="#DD8452", edgecolor="black", linewidth=0.5, hatch="//")

    axes.set_xticks(positions)
    axes.set_xticklabels(series, rotation=20, ha="right")
    axes.set_ylabel("F1 (hallucinated = positive class)")
    axes.set_ylim(0, 1)
    axes.set_title("Hallucination detection quality per judge")
    axes.grid(alpha=0.3, axis="y")
    axes.legend(loc="lower right", fontsize=8)
    return _save(figure, _figure_path(out_dir, "reliability"))


def plot_tradeoff(
    reports: dict[str, JudgeReport],
    costs: CostModel,
    out_dir: Path,
    *,
    samples: int = 1000,
) -> Path:
    """Report Figure 6.3: quality against latency, sized by cost — the "when is an SLM
    worth it" figure.

    One point per judge. The y axis is the held-out F1 where there is one, because the
    in-sample number would flatter every judge by a different amount and distort exactly the
    comparison this figure exists to make.
    """
    figure, axes = plt.subplots(figsize=(7, 5))
    markers = ("o", "s", "^", "D", "v", "P")

    plotted = False
    for index, name in enumerate(sorted(reports)):
        report = reports[name]
        quality = report.holdout.metrics.f1 if report.holdout else (
            report.best.f1 if report.best else None
        )
        if quality is None or report.latency_median_ms is None:
            continue
        cost = costs.estimate(report, samples)
        axes.scatter(
            report.latency_median_ms,
            quality,
            s=120,
            marker=markers[index % len(markers)],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            label=f"{name} (${cost:.2f}/1k)",
        )
        axes.annotate(
            name,
            (report.latency_median_ms, quality),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
        plotted = True

    axes.set_xlabel("Median latency per evaluation (ms) — lower is better")
    axes.set_ylabel("F1 at the held-out threshold — higher is better")
    axes.set_title(f"Quality against latency (legend shows USD per {samples} evaluations)")
    axes.set_ylim(0, 1.05)
    axes.grid(alpha=0.3)
    if plotted:
        axes.legend(loc="lower left", fontsize=8)
        # The upper-left corner is the one worth being in; naming it saves a paragraph.
        axes.text(
            0.02, 0.97, "better", transform=axes.transAxes, fontsize=9,
            va="top", ha="left", style="italic", color="gray",
        )
    return _save(figure, _figure_path(out_dir, "tradeoff"))


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
    ablations: Sequence[PrivacyAblation] = (),
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

    sections += [
        "## Detection quality at a held-out threshold",
        "",
        "The table above chooses each threshold on the same rows it reports, so its F1 is an "
        "upper bound. Here the threshold is chosen on half the labeled rows and measured on "
        "the other half, split by a hash of the sample id so every judge is measured on the "
        "same held-out samples. **These are the numbers to quote as the result**; the "
        "difference between the two tables is how much the in-sample threshold flatters the "
        "judge.",
        "",
        _markdown_table(
            ["Judge", "Fit n", "Test n", "t", "P", "R", "F1", "Accuracy", "Kappa"],
            [
                [
                    report.series,
                    str(report.holdout.selection_size) if report.holdout else "n/a",
                    str(report.holdout.reporting_size) if report.holdout else "n/a",
                    _format(report.holdout.threshold, 2) if report.holdout else "n/a",
                    _format(report.holdout.metrics.precision) if report.holdout else "n/a",
                    _format(report.holdout.metrics.recall) if report.holdout else "n/a",
                    _format(report.holdout.metrics.f1) if report.holdout else "n/a",
                    _format(report.holdout.metrics.accuracy) if report.holdout else "n/a",
                    _format(report.holdout.metrics.cohen_kappa) if report.holdout else "n/a",
                ]
                for report in reports.values()
            ],
        ),
        "",
        "Kappa here is agreement with the **human labels**, corrected for chance. On an "
        "unbalanced set raw accuracy flatters a judge that always answers the same way; "
        "kappa does not.",
        "",
        "`n/a` means the split could not measure it — one of the two halves held only one "
        "class, so no threshold is distinguishable from any other. It does not mean the "
        "judge scored zero. Increase the sample size or stratify the draw.",
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

    sections += ["## Privacy ablation: what masking costs", ""]
    if ablations:
        sections += [
            _markdown_table(
                ["Judge", "Model", "Shared samples", "F1 masked", "F1 unmasked", "ΔF1", "ΔAcc"],
                [
                    [
                        item.judge,
                        item.model,
                        str(item.shared_samples),
                        _format(item.masked_f1),
                        _format(item.unmasked_f1),
                        _format(item.delta_f1),
                        _format(item.delta_accuracy),
                    ]
                    for item in ablations
                ],
            ),
            "",
            "Δ is masked minus unmasked, so a negative value is accuracy given up in exchange "
            "for the privacy guarantee. Both sides use each run's held-out threshold.",
            "",
        ]
    else:
        sections += [
            "No judge was run both with and without masking, so the cost of the privacy "
            "layer is not measured here. Run the same dataset twice — `--privacy-mode mask` "
            "and `--privacy-mode off`, into different `--out` directories — and analyze both.",
            "",
        ]

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
    ablations = analyze_privacy_ablation(frame, reports)
    figures = [
        plot_reliability(reports, out_dir),
        plot_roc_curves(frame, reports, out_dir),
        plot_tradeoff(reports, cost_model, out_dir),
        plot_score_distributions(frame, out_dir),
        plot_latency_box(frame, out_dir),
    ]

    summary_path = out_dir / "summary.md"
    summary_path.write_text(
        render_summary(frame, reports, agreements, cost_model, figures, ablations),
        encoding="utf-8",
    )
    results_path = export_results(reports, agreements, cost_model, out_dir, ablations)
    return {
        "summary_path": summary_path,
        "results_path": results_path,
        "figures": figures,
        "reports": reports,
        "agreements": agreements,
        "ablations": ablations,
    }


def export_results(
    reports: dict[str, JudgeReport],
    agreements: Sequence[AgreementReport],
    costs: CostModel,
    out_dir: Path,
    ablations: Sequence[PrivacyAblation] = (),
) -> Path:
    """Write results.json: every number in the summary, machine-readable.

    Report prose has to quote exact figures, and re-typing them from a Markdown table is how
    a report ends up with a number that no longer matches the run that produced it. This is
    the file to read them from.
    """
    payload = {
        "judges": {
            name: {
                "judge": report.judge,
                "model": report.model,
                "datasets": list(report.datasets),
                "coverage": {
                    "scored": report.scored,
                    "unscored": report.unscored,
                    "unlabeled": report.unlabeled,
                    "failed": report.failed,
                    "positives": report.positives,
                    "negatives": report.negatives,
                },
                "in_sample": _metrics_payload(report.best),
                "held_out": (
                    {
                        "threshold": report.holdout.threshold,
                        "selection_size": report.holdout.selection_size,
                        "reporting_size": report.holdout.reporting_size,
                        **(_metrics_payload(report.holdout.metrics) or {}),
                    }
                    if report.holdout
                    else None
                ),
                "roc_auc": report.roc_auc,
                "latency_median_ms": report.latency_median_ms,
                "latency_p95_ms": report.latency_p95_ms,
                "mean_tokens": report.mean_tokens,
                "estimated_usd_per_1k_evals": costs.estimate(report, 1000),
            }
            for name, report in reports.items()
        },
        "privacy_ablation": [
            {
                "judge": item.judge,
                "model": item.model,
                "shared_samples": item.shared_samples,
                "masked_f1": item.masked_f1,
                "unmasked_f1": item.unmasked_f1,
                "delta_f1": item.delta_f1,
                "delta_accuracy": item.delta_accuracy,
            }
            for item in ablations
        ],
        "agreement": [
            {
                "judge_a": item.judge_a,
                "judge_b": item.judge_b,
                "overlap": item.overlap,
                "pearson": item.pearson,
                "spearman": item.spearman,
                "cohen_kappa": item.cohen_kappa,
            }
            for item in agreements
        ],
    }
    path = out_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _metrics_payload(metrics: ThresholdMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "threshold": metrics.threshold,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "true_negatives": metrics.true_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "cohen_kappa": metrics.cohen_kappa,
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
