"""Calibration helpers for matches_ai_pile (1% FPR on human reference)."""

from __future__ import annotations

from typing import Any


def threshold_at_fpr(human_scores: list[float], fpr: float = 0.01) -> float:
    if not human_scores:
        return 1.0
    ordered = sorted(human_scores, reverse=True)
    k = max(0, min(len(ordered) - 1, int(len(ordered) * fpr)))
    return float(ordered[k])


def human_percentile(score: float, human_scores: list[float]) -> float:
    if not human_scores:
        return 0.0
    n = sum(1 for s in human_scores if score > s)
    return 100.0 * n / len(human_scores)


def calibration_record(human_scores: list[float], fpr: float = 0.01) -> dict[str, Any]:
    return {
        "fpr": fpr,
        "threshold": threshold_at_fpr(human_scores, fpr),
        "n_human": len(human_scores),
        "label": "matches_ai_pile",
    }
