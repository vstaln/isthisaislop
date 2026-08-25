"""Confound probes: deliberately dumb models that must FAIL on a healthy corpus.

Every probe that succeeds is a shortcut the real detector is also free to take, which means the
headline AUROC is measuring the shortcut instead of the writing.

    uv run python scripts/corpus_probes.py --parquet data/v2/v2_train.parquet.labeled.parquet
    uv run python scripts/corpus_probes.py --hf vstalingrady/itais/v2_train_labeled.parquet

Writes the results to artifacts/corpus_probes.json.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HF_TEMPLATE = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"
THRESHOLDS = {"register_only": 0.55, "length_only": 0.55}


def auroc(scores, labels) -> float:
    labels = np.asarray(labels)
    pos, neg = labels.sum(), (1 - labels).sum()
    if not pos or not neg:
        return float("nan")
    ranks = pd.Series(np.asarray(scores)).rank().values
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def two_sided(value: float) -> float:
    """A probe that predicts the label backwards is just as much of a shortcut."""
    return max(value, 1.0 - value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path)
    ap.add_argument("--hf", help="repo/file, e.g. vstalingrady/itais/v2_train_labeled.parquet")
    ap.add_argument("--out", type=Path, default=Path("artifacts/corpus_probes.json"))
    args = ap.parse_args()

    source = args.parquet
    if args.hf:
        repo, name = args.hf.rsplit("/", 1)
        source = Path("/tmp") / name
        if not source.exists():
            urllib.request.urlretrieve(HF_TEMPLATE.format(repo=repo, name=name), source)
    if not source or not source.exists():
        raise SystemExit("pass --parquet or --hf")

    df = pd.read_parquet(source)
    df["n_words"] = df.text.str.split().str.len()

    per_register = df.groupby("register").label.agg(n="size", ai_frac="mean")
    single = per_register[(per_register.ai_frac <= 0.02) | (per_register.ai_frac >= 0.98)]
    df["p_register"] = df.register.map(per_register.ai_frac)

    report = {
        "source": str(source),
        "rows": len(df),
        "registers": len(per_register),
        "single_label_registers": len(single),
        "docs_in_single_label_registers": int(single.n.sum()),
        "share_single_label": round(float(single.n.sum() / len(df)), 3),
        "probes": {
            "register_only": round(two_sided(auroc(df.p_register.values, df.label.values)), 3),
            "length_only": round(two_sided(auroc(df.n_words.values, df.label.values)), 3),
        },
        "median_words": {"human": float(df[df.label == 0].n_words.median()),
                         "ai": float(df[df.label == 1].n_words.median())},
        "mixed_registers": per_register[(per_register.ai_frac > 0.02)
                                        & (per_register.ai_frac < 0.98)].round(3).to_dict("index"),
        "top_generators": df[df.label == 1].generator.value_counts().head(10).to_dict(),
        "thresholds": THRESHOLDS,
    }
    report["verdict"] = {name: ("PASS" if value <= THRESHOLDS[name] else "FAIL")
                         for name, value in report["probes"].items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all(v == "PASS" for v in report["verdict"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
