"""AUROC and per-register calibration.

The first test is the regression guard that matters: the AUROC this module
replaced summed *input positions* rather than score ranks, so it returned 1.0 for
any input whose positive rows happened to sit at the end — which is how every
register-vs-register number in the v1 post-mortem came out at exactly 1.00.
"""

from __future__ import annotations

from math import isnan

from slopdet.metrics import (
    auroc,
    cross_register_auroc,
    gated_auroc,
    per_register_metrics,
    register_thresholds,
    tpr_at_fpr,
)

LABELS = [0, 0, 0, 1, 1, 1]


def test_auroc_reads_the_scores_not_the_row_order():
    assert auroc([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], LABELS) == 1.0
    assert auroc([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], LABELS) == 0.0
    assert auroc([0.5, 0.9, 0.1, 0.4, 0.8, 0.2], LABELS) == 0.4444444444444444


def test_constant_scores_are_chance_not_perfect():
    assert auroc([0.5] * 6, LABELS) == 0.5


def test_ties_get_the_average_rank():
    # One human tied with two AI docs: those two comparisons score half each, so
    # 8 of 9 human/AI pairs are ordered correctly. Matches sklearn's roc_auc_score.
    assert auroc([0.1, 0.2, 0.5, 0.5, 0.5, 0.9], LABELS) == 0.8888888888888888


def test_auroc_is_nan_without_both_classes():
    assert isnan(auroc([0.1, 0.2], [0, 0]))
    assert isnan(auroc([], []))


def test_gate_hides_slices_too_small_to_mean_anything():
    scores = [i / 300 for i in range(300)]
    labels = [0] * 150 + [1] * 150
    assert gated_auroc(scores, labels) == 1.0
    assert isnan(gated_auroc(scores[:100], labels[:50] + labels[150:200]))
    assert isnan(gated_auroc(scores, [0] * 290 + [1] * 10))


def test_threshold_lets_one_percent_of_humans_through():
    humans = [i / 1000 for i in range(1000)]
    ai = [1.5] * 100
    tpr, threshold = tpr_at_fpr(humans + ai, [0] * 1000 + [1] * 100, fpr=0.01)
    assert tpr == 1.0
    assert sum(1 for s in humans if s > threshold) / len(humans) <= 0.01


def test_thresholds_are_per_register():
    scores = [0.1, 0.2, 0.9] * 2 + [0.6, 0.7, 0.95] * 2
    labels = [0, 0, 1] * 2 + [0, 0, 1] * 2
    registers = ["blogs"] * 6 + ["coai"] * 6
    thresholds = register_thresholds(scores, labels, registers)
    assert set(thresholds) == {"blogs", "coai", "all"}
    assert thresholds["blogs"] < thresholds["coai"]


def test_per_register_records_where_its_threshold_came_from():
    scores = [0.1, 0.9] * 4
    labels = [0, 1] * 4
    registers = ["blogs"] * 4 + ["coai"] * 4
    metrics = per_register_metrics(scores, labels, registers, {"blogs": 0.5})
    assert metrics["blogs"]["threshold_source"] == "cal"
    assert metrics["coai"]["threshold_source"] == "val"
    assert metrics["blogs"]["n"] == 4 and metrics["blogs"]["n_ai"] == 2
    assert metrics["all"]["n"] == 8
    assert "note" in metrics["all"]


def test_single_class_register_reports_nan_tpr():
    metrics = per_register_metrics([0.2, 0.3], [0, 0], ["gutenberg", "gutenberg"])
    assert isnan(metrics["gutenberg"]["tpr_at_1pct_fpr"])
    assert isnan(metrics["gutenberg"]["auroc_raw"])


def test_cross_register_matrix_uses_only_single_class_registers():
    scores = [0.1, 0.2, 0.8, 0.9, 0.4, 0.6]
    labels = [0, 0, 1, 1, 0, 1]
    registers = ["gutenberg", "gutenberg", "m4_reddit", "m4_reddit", "coai", "coai"]
    matrix = cross_register_auroc(scores, labels, registers)
    # coai holds both classes, so it is neither a human nor an AI axis.
    assert set(matrix) == {"gutenberg_vs_m4_reddit"}
    assert matrix["gutenberg_vs_m4_reddit"]["auroc"] == 1.0
    assert matrix["gutenberg_vs_m4_reddit"]["n"] == 4
