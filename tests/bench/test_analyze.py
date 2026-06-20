"""Every expected number here is derived by hand from tests/data/bench_two_judges.jsonl.

The fixture is deliberately tiny so each metric can be recomputed on paper:

    judge  sample  label         faithfulness   latency_ms
    slm    s1      not-halluc.   1.0            10
    slm    s2      not-halluc.   0.8            20
    slm    s3      hallucinated  0.4            30
    slm    s4      hallucinated  0.0            40
    slm    s5      hallucinated  null (unscored) 50
    cloud  s1      not-halluc.   0.9            100
    cloud  s2      not-halluc.   0.7            200
    cloud  s3      hallucinated  0.5            300
    cloud  s4      hallucinated  0.1            400
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from slm_rag_eval.bench.analyze import (
    THRESHOLDS,
    CostModel,
    analyze,
    analyze_agreement,
    analyze_judge,
    load_rows,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "bench_two_judges.jsonl"
EXACT = 1e-9


def _reports() -> dict[str, object]:
    frame = load_rows([FIXTURE])
    return {
        judge: analyze_judge(frame[frame["judge"] == judge]) for judge in ("slm", "cloud")
    }


def test_threshold_sweep_covers_zero_to_one_in_steps_of_five_hundredths() -> None:
    assert THRESHOLDS[0] == 0.0
    assert THRESHOLDS[-1] == 1.0
    assert len(THRESHOLDS) == 21
    assert THRESHOLDS[1] == 0.05


def test_unscored_rows_are_counted_separately_and_excluded() -> None:
    report = _reports()["slm"]

    assert report.scored == 4
    assert report.unscored == 1  # s5 has faithfulness null
    assert report.positives == 2  # only the scored rows count toward the sweep
    assert report.negatives == 2


def test_slm_sweep_matches_hand_computed_confusion_counts() -> None:
    sweep = {metrics.threshold: metrics for metrics in _reports()["slm"].sweep}

    # t=0.00 predicts nothing: every hallucination is missed.
    at_zero = sweep[0.0]
    assert (at_zero.true_positives, at_zero.false_positives) == (0, 0)
    assert (at_zero.false_negatives, at_zero.true_negatives) == (2, 2)
    assert at_zero.f1 == pytest.approx(0.0, abs=EXACT)
    assert at_zero.balanced_accuracy == pytest.approx(0.5, abs=EXACT)

    # t=0.05 catches only s4 (0.0): precision 1, recall 1/2.
    low = sweep[0.05]
    assert (low.true_positives, low.false_positives) == (1, 0)
    assert low.precision == pytest.approx(1.0, abs=EXACT)
    assert low.recall == pytest.approx(0.5, abs=EXACT)
    assert low.f1 == pytest.approx(2 / 3, abs=EXACT)
    assert low.balanced_accuracy == pytest.approx(0.75, abs=EXACT)

    # t=0.45 catches s3 and s4 and nothing else: a perfect split.
    perfect = sweep[0.45]
    assert (perfect.true_positives, perfect.false_positives) == (2, 0)
    assert perfect.f1 == pytest.approx(1.0, abs=EXACT)
    assert perfect.balanced_accuracy == pytest.approx(1.0, abs=EXACT)

    # t=0.85 also flags s2 (0.8), which is not hallucinated.
    over = sweep[0.85]
    assert (over.true_positives, over.false_positives) == (2, 1)
    assert over.precision == pytest.approx(2 / 3, abs=EXACT)
    assert over.f1 == pytest.approx(0.8, abs=EXACT)


def test_best_threshold_is_the_lowest_one_reaching_the_top_f1() -> None:
    reports = _reports()

    assert reports["slm"].best.threshold == pytest.approx(0.45, abs=EXACT)
    assert reports["slm"].best.f1 == pytest.approx(1.0, abs=EXACT)
    # The cloud judge separates at 0.5, so its own best threshold is one step higher.
    assert reports["cloud"].best.threshold == pytest.approx(0.55, abs=EXACT)
    assert reports["cloud"].best.f1 == pytest.approx(1.0, abs=EXACT)


def test_roc_auc_is_perfect_when_scores_separate_the_classes() -> None:
    reports = _reports()

    assert reports["slm"].roc_auc == pytest.approx(1.0, abs=EXACT)
    assert reports["cloud"].roc_auc == pytest.approx(1.0, abs=EXACT)


def test_latency_and_token_efficiency_per_judge() -> None:
    reports = _reports()

    # slm latencies 10,20,30,40,50 -> median 30; p95 interpolates at index 0.95*4 = 3.8.
    assert reports["slm"].latency_median_ms == pytest.approx(30.0, abs=EXACT)
    assert reports["slm"].latency_p95_ms == pytest.approx(48.0, abs=EXACT)
    # cloud latencies 100,200,300,400 -> median 250; index 0.95*3 = 2.85.
    assert reports["cloud"].latency_median_ms == pytest.approx(250.0, abs=EXACT)
    assert reports["cloud"].latency_p95_ms == pytest.approx(385.0, abs=EXACT)

    assert reports["slm"].mean_tokens["total_tokens"] == pytest.approx(120.0, abs=EXACT)
    assert reports["cloud"].mean_tokens["prompt_tokens"] == pytest.approx(200.0, abs=EXACT)


def test_agreement_over_the_samples_both_judges_scored() -> None:
    frame = load_rows([FIXTURE])
    reports = {judge: analyze_judge(frame[frame["judge"] == judge]) for judge in ("cloud", "slm")}

    agreements = analyze_agreement(frame, reports)

    assert len(agreements) == 1
    agreement = agreements[0]
    assert (agreement.judge_a, agreement.judge_b) == ("cloud", "slm")
    assert agreement.overlap == 4  # s5 is slm-only and unscored

    # Pearson on (1.0,0.8,0.4,0.0) vs (0.9,0.7,0.5,0.1): cov 0.45, variances 0.59 and 0.35.
    assert agreement.pearson == pytest.approx(0.45 / math.sqrt(0.59 * 0.35), abs=EXACT)
    # Both judges rank the four samples identically, so the rank correlation is exactly 1.
    assert agreement.spearman == pytest.approx(1.0, abs=EXACT)
    # At 0.45 and 0.55 respectively both judges call s3 and s4 hallucinated and agree fully.
    assert agreement.cohen_kappa == pytest.approx(1.0, abs=EXACT)


def test_cost_model_prices_the_cloud_judge_and_zeroes_the_local_one() -> None:
    reports = _reports()
    costs = CostModel(cloud_input_per_1m=5.0, cloud_output_per_1m=15.0)

    # 4 cloud samples x (200 prompt, 40 completion) = (800 * 5 + 160 * 15) / 1e6.
    assert costs.estimate(reports["cloud"], 4) == pytest.approx(0.0064, abs=EXACT)
    assert costs.estimate(reports["slm"], 5) == pytest.approx(0.0, abs=EXACT)


def test_analyze_writes_every_output_file(tmp_path: Path) -> None:
    outcome = analyze([FIXTURE], tmp_path, costs=CostModel(5.0, 15.0))

    summary = tmp_path / "summary.md"
    assert summary.exists()
    assert {path.name for path in outcome["figures"]} == {
        "reliability.png",
        "roc_curves.png",
        "tradeoff.png",
        "score_distributions.png",
        "latency_box.png",
    }
    for figure in outcome["figures"]:
        assert figure.exists()
        assert figure.stat().st_size > 0

    text = summary.read_text()
    assert "# Benchmark summary" in text
    assert "Detection quality at the best-F1 threshold" in text
    assert "Judge agreement" in text
    assert "Efficiency and cost" in text
    assert "![roc_curves](roc_curves.png)" in text
    # The unscored row is reported rather than hidden.
    assert "Unscored" in text


def test_load_rows_rejects_empty_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No benchmark rows"):
        load_rows([empty])
