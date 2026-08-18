#!/usr/bin/env python3
"""Train the CPU matches_ai_pile scorer on coai. No GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from colab_pipeline import load_coai
from slopdet.calibrate import calibration_record
from slopdet.ontology import load_ontology
from slopdet.scorer import N_EXTRA, default_bundle_path, featurize


def main() -> None:
    onto = load_ontology(ROOT / "ontology")
    ids = [p.id for p in onto.enabled_patterns()]
    data = load_coai(ROOT / "data")
    rng = np.random.default_rng(0)

    def matrix(docs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([np.array(featurize(d["text"], onto, ids), dtype=np.float32) for d in docs])
        y = np.array([int(d["label"]) for d in docs], dtype=np.int64)
        return x, y

    train = data["train"]
    if "test" in data:
        test = data["test"]
    else:
        idx = rng.permutation(len(train))
        cut = int(0.85 * len(train))
        test = [train[i] for i in idx[cut:]]
        train = [train[i] for i in idx[:cut]]

    # Cap so a laptop finishes in minutes; 8k is already 400x the seed bundle.
    if len(train) > 8000:
        pick = rng.choice(len(train), size=8000, replace=False)
        train = [train[int(i)] for i in pick]
    if len(test) > 2000:
        pick = rng.choice(len(test), size=2000, replace=False)
        test = [test[int(i)] for i in pick]

    print("featurizing", len(train), "train", len(test), "eval")
    x_train, y_train = matrix(train)
    x_test, y_test = matrix(test)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(xs, y_train)
    probs = clf.predict_proba(scaler.transform(x_test))[:, 1]
    auc = float(roc_auc_score(y_test, probs))
    acc = float(((probs >= 0.5).astype(int) == y_test).mean())
    print("auc", round(auc, 4), "acc", round(acc, 4))

    human_scores = clf.predict_proba(scaler.transform(x_test[y_test == 0]))[:, 1].tolist()
    calib = calibration_record(human_scores, 0.01)
    calib["human_scores"] = human_scores
    path = default_bundle_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pattern_ids": ids,
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "calibration": calib,
        "ontology_sha256": onto.sha256,
        "n_docs": len(train),
        "n_eval": len(test),
        "auc_eval": auc,
        "acc_eval": acc,
        "trained_on": [
            "coai/ai-text-detection-training 8k sample (arxiv abstracts vs LLM paraphrases)"
        ],
        "never_trained_on": ["coai held-out eval slice"],
        "n_features": len(ids) + N_EXTRA,
    }
    path.write_text(json.dumps(bundle), encoding="utf-8")
    print("wrote", path, "threshold", round(calib["threshold"], 4))


if __name__ == "__main__":
    main()
