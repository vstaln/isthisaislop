"""Calibration helpers for matches_ai_pile (1% FPR on human reference)."""

from __future__ import annotations

from typing import Any


def threshold_at_fpr(human_scores: list[float], fpr: float = 0.01) -> float:
    """Threshold where ~fpr of humans score strictly above it (descending)."""
    if not human_scores:
        return 1.0
    ordered = sorted(human_scores, reverse=True)
    k = max(0, min(len(ordered) - 1, int(len(ordered) * fpr)))
    return float(ordered[k])


# Registry of registers that may appear in training parquets (K3 gate).
# v1: coai/pile/storyscope/gutenberg/blogs/scp/writingprompts.
# v2 adds prefixed families (raid_*, m4_*, hc3_*, wiki_intro*, beemo*,
# semeval_mixed) — matched by prefix so new domains need no code change.
ALLOWED_REGISTERS = frozenset({"coai", "pile", "storyscope", "gutenberg", "blogs",
                               "scp", "writingprompts", "smoke"})
ALLOWED_REGISTER_PREFIXES = ("raid_", "m4_", "hc3_", "wiki_intro", "beemo",
                             "semeval_mixed", "fictpair")


def register_allowed(reg: str) -> bool:
    return reg in ALLOWED_REGISTERS or reg.startswith(ALLOWED_REGISTER_PREFIXES)


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
