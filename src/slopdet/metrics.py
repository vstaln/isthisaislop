"""Per-register evaluation at a fixed FPR — one implementation, every caller.

The trainer, the post-training evaluator and any future report all read their
numbers from here, so a metric can never mean two things in two places. The
threshold is always calibrated per register (docs/HANDOFF.md §5); a single global
cut fitted on one register does not hold on another.
"""

from __future__ import annotations

from .calibrate import threshold_at_fpr

MIN_N = 200
MIN_POS = 20


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank AUROC with tie correction. nan when one class is missing.

    Ties get the average rank, so a model that outputs one constant score scores
    0.5 rather than whatever the input happened to be ordered by.
    """
    if not labels:
        return float("nan")
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    ranked = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        rank_sum += avg_rank * sum(label for _, label in ranked[i:j + 1])
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def gated_auroc(scores: list[float], labels: list[int],
                min_n: int = MIN_N, min_pos: int = MIN_POS) -> float:
    """AUROC, or nan when the slice is too small for the number to mean anything."""
    if len(labels) < min_n:
        return float("nan")
    if sum(labels) < min_pos or sum(1 for y in labels if y == 0) < min_pos:
        return float("nan")
    return auroc(scores, labels)


def tpr_at_fpr(scores: list[float], labels: list[int],
               fpr: float = 0.01) -> tuple[float, float]:
    """(TPR, threshold) at the given human false-positive rate."""
    human = [s for s, y in zip(scores, labels) if y == 0]
    if not human:
        return float("nan"), float("nan")
    threshold = threshold_at_fpr(human, fpr)
    ai = [s for s, y in zip(scores, labels) if y == 1]
    if not ai:
        return float("nan"), threshold
    return sum(1 for s in ai if s > threshold) / len(ai), threshold


def register_thresholds(cal_scores: list[float], cal_labels: list[int],
                        cal_registers: list[str], fpr: float = 0.01) -> dict[str, float]:
    """Operating threshold per register, fitted on the calibration slice only.

    Includes an 'all' entry as the fallback for a register with no cal rows. Keep
    cal disjoint from val or the reported FPR is the one you fitted, not one you
    measured.
    """
    if not cal_scores:
        return {}
    out = {}
    for register in sorted(set(cal_registers)):
        idx = [i for i, r in enumerate(cal_registers) if r == register]
        out[register] = tpr_at_fpr([cal_scores[i] for i in idx],
                                   [cal_labels[i] for i in idx], fpr)[1]
    out["all"] = tpr_at_fpr(cal_scores, cal_labels, fpr)[1]
    return out


def per_register_metrics(scores: list[float], labels: list[int], registers: list[str],
                         thresholds: dict[str, float] | None = None,
                         fpr: float = 0.01) -> dict[str, dict]:
    """AUROC (gated + raw), TPR at the calibrated threshold, and n, per register.

    `thresholds` comes from `register_thresholds` on the calibration slice. A
    register missing from it falls back to a threshold fitted on its own rows,
    which is circular and is recorded as `threshold_source: "val"` so a reader can
    see the difference.
    """
    thresholds = thresholds or {}
    out: dict[str, dict] = {}
    for register in sorted(set(registers)) + ["all"]:
        if register == "all":
            idx = list(range(len(labels)))
        else:
            idx = [i for i, r in enumerate(registers) if r == register]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        known = register in thresholds
        threshold = thresholds[register] if known else tpr_at_fpr(s, y, fpr)[1]
        n_ai = sum(1 for label in y if label == 1)
        n_human = len(y) - n_ai
        tpr = (sum(1 for sc, label in zip(s, y) if label == 1 and sc > threshold) / n_ai
               if n_ai and n_human else float("nan"))
        out[register] = {
            "n": len(idx),
            "n_ai": n_ai,
            "auroc": gated_auroc(s, y),
            "auroc_raw": auroc(s, y),
            "tpr_at_1pct_fpr": tpr,
            "threshold": threshold,
            "threshold_source": "cal" if known else "val",
        }
    out["all"]["note"] = ("'all' mixes registers of very different size; read the "
                          "per-register rows, not this one")
    return out


def cross_register_auroc(scores: list[float], labels: list[int],
                         registers: list[str]) -> dict[str, dict]:
    """AUROC for every (human-only register) x (AI-only register) pair.

    A detector that has learnt register or era rather than provenance scores near
    1.0 on every cross pair while scoring near chance inside a register that holds
    both classes. Register roles are derived from the labels actually present.
    """
    seen: dict[str, set[int]] = {}
    for label, register in zip(labels, registers):
        seen.setdefault(register, set()).add(label)
    human_regs = sorted(r for r, ys in seen.items() if ys == {0})
    ai_regs = sorted(r for r, ys in seen.items() if ys == {1})

    out: dict[str, dict] = {}
    for human in human_regs:
        h_scores = [s for s, r in zip(scores, registers) if r == human]
        for ai in ai_regs:
            a_scores = [s for s, r in zip(scores, registers) if r == ai]
            if not h_scores or not a_scores:
                continue
            paired = h_scores + a_scores
            paired_labels = [0] * len(h_scores) + [1] * len(a_scores)
            out[f"{human}_vs_{ai}"] = {"n": len(paired),
                                       "auroc": auroc(paired, paired_labels)}
    return out
