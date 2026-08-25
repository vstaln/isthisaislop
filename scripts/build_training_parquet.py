#!/usr/bin/env python3
"""Combine all labeled spans parquets into one training parquet with a register column.

The fine-tune script (scripts/fine_tune_lfm.py --spans-parquet) expects rows
with text / label (0|1) / register / spans. Each spans_*.parquet has
text / pile / slop_tags / human_tags / spans / model but no register — this
script assigns the register from the corpus name and writes the merged file.

Usage:
  uv run python scripts/build_training_parquet.py [--out data/training/train_all.parquet]

The storyscope corpus is AI fiction (register 'storyscope'), gutenberg/blogs/scp
are human (registers 'gutenberg','blogs','scp'), coai is balanced academic
(register 'coai'). Pile 0 = human, 1 = AI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from slopdet.corpus import coerce_spans  # noqa: E402
from slopdet.labels import parse_label  # noqa: E402


def load_spans(path: Path, register: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rows = []
    for rec in df.to_dict("records"):
        # `rec.get("spans") or []` used to live here and raised
        # "truth value of an array with more than one element is ambiguous" on
        # every multi-span row, because pandas hands the column back as a numpy
        # array. coerce_spans is the one tolerant reader, shared with the trainer.
        spans = coerce_spans(rec.get("spans"))
        rows.append(
            {
                "text": rec["text"],
                # default 1 (AI) for the AI-only corpora, 0 (human) otherwise;
                # coai has an explicit label column so its default never applies
                "label": parse_label(rec, default=1 if register == "storyscope" else 0),
                "register": register,
                "spans": [
                    {
                        "lane": s.get("lane"),
                        "start": int(s["start"]),
                        "end": int(s["end"]),
                    }
                    for s in spans
                    if s.get("lane") and s.get("start") is not None and s.get("end") is not None
                ],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "training" / "train_all.parquet")
    args = ap.parse_args()

    corpora = [
        ("data/training/spans_coai_train.parquet", "coai"),
        ("data/training/spans_storyscope_train.parquet", "storyscope"),
        ("data/training/spans_gutenberg_train.parquet", "gutenberg"),
        ("data/training/spans_blogs_train.parquet", "blogs"),
        ("data/training/spans_scp_train.parquet", "scp"),
        # artem9k ai-text-detection-pile: real internet text, doc labels from
        # source (human/ai); spans labeled by scripts/label_artem9k.py.
        # register "pile" distinguishes real-world web text from curated corpora.
        ("data/training/spans_artem9k_train.parquet", "pile"),
        # writingprompts: modern human short fiction (100k) — breaks the
        # storyscope=AI / gutenberg=human fiction confound (K3 fix). Same human label.
        ("data/training/spans_writingprompts_train.parquet", "writingprompts"),
    ]
    frames = []
    for rel, register in corpora:
        p = ROOT / rel
        if not p.exists():
            print(f"skip {rel} (missing)", flush=True)
            continue
        df = load_spans(p, register)
        frames.append(df)
        print(f"{rel}: {len(df)} rows, register={register}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    print(f"\ntotal {len(out)} rows", flush=True)
    print(f"pile: {out['label'].value_counts().to_dict()}", flush=True)
    print(f"register: {out['register'].value_counts().to_dict()}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
