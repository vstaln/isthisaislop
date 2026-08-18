"""CPU pile-resemblance scorer. Numpy at inference; sklearn only to train."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from slopdet.calibrate import human_percentile
from slopdet.construction import construction_stats
from slopdet.ontology import Ontology, load_ontology
from slopdet.weaklabel import label_text

CONSTRUCTION_KEYS = (
    "burstiness",
    "evenness",
    "recap_closure",
    "over_explain",
    "portability",
)
N_EXTRA = len(CONSTRUCTION_KEYS) + 1  # + n_words/1000


def default_bundle_path() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "sklearn_bundle.json"


def featurize(
    text: str,
    ontology: Ontology,
    pattern_ids: list[str],
) -> list[float]:
    hits = label_text(text, ontology)
    counts = dict.fromkeys(pattern_ids, 0.0)
    for hit in hits:
        if hit["id"] in counts:
            counts[hit["id"]] += 1.0
    stats = construction_stats(text)
    extra = [float(stats.get(k) or 0.0) for k in CONSTRUCTION_KEYS]
    extra.append(float(stats.get("n_words") or 0.0) / 1000.0)
    return [counts[pid] for pid in pattern_ids] + extra


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@lru_cache(maxsize=1)
def load_bundle(path: Path | None = None) -> dict[str, Any]:
    path = path or default_bundle_path()
    return json.loads(path.read_text(encoding="utf-8"))


def score_vector(vec: list[float], bundle: dict[str, Any]) -> float:
    mean = bundle["scaler_mean"]
    scale = bundle["scaler_scale"]
    coef = bundle["coef"]
    if len(vec) != len(coef):
        raise ValueError(f"feature dim {len(vec)} != coef dim {len(coef)}")
    acc = float(bundle["intercept"])
    for v, m, s, c in zip(vec, mean, scale, coef):
        acc += c * ((v - m) / (s if s else 1.0))
    return _sigmoid(acc)


def score_text(
    text: str,
    *,
    ontology: Ontology | None = None,
    bundle: dict[str, Any] | None = None,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    bundle = bundle or load_bundle(bundle_path)
    onto = ontology or load_ontology()
    vec = featurize(text, onto, bundle["pattern_ids"])
    score = score_vector(vec, bundle)
    human_scores = bundle.get("calibration", {}).get("human_scores") or []
    pct = human_percentile(score, human_scores)
    return {
        "label": "matches_ai_pile",
        "score": score,
        "human_percentile": pct,
        "text": (
            f"Resembles the AI pile more than {pct:.0f}% of human reference texts."
        ),
        "trained_on": bundle.get("trained_on"),
    }
