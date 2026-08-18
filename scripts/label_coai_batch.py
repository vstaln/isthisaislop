#!/usr/bin/env python3
"""Label the full coai train slice (62k docs) with deterministic spans.

One row per doc: pile target, lean, slop/human tag lists, verbatim spans.
Multiprocessed; explain(sentences=False) skips the per-sentence pass.
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from slopdet.explain import explain  # noqa: E402


def spans_for(out: dict) -> list[dict]:
    spans = []
    for hit in (out.get("why_slop") or []) + (out.get("why_human") or []):
        if not hit.get("quote"):
            continue
        spans.append(
            {
                "id": hit["id"],
                "lane": hit.get("lane", "style"),
                "lean": hit.get("lean", "slop"),
                "start": hit.get("start"),
                "end": hit.get("end"),
                "quote": hit["quote"],
            }
        )
    return spans


def label_row(args: tuple) -> dict:
    idx, text, label = args
    out = explain(text, sentences=False)
    return {
        "idx": idx,
        "text": text,
        "pile": int(label),
        "slop_tags": [h["id"] for h in out["why_slop"]],
        "human_tags": [h["id"] for h in out["why_human"]],
        "spans": json.dumps(spans_for(out), ensure_ascii=False),
    }


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "coai_train.parquet")
    print(f"coai train: {len(df)} docs", flush=True)
    jobs = [(i, str(text), int(label)) for i, (text, label) in enumerate(zip(df["text"], df["label"]))]
    t0 = time.time()
    with Pool() as pool:
        rows = list(pool.imap_unordered(label_row, jobs, chunksize=64))
    print(f"labeled {len(rows)} in {time.time() - t0:.0f}s", flush=True)

    out_df = pd.DataFrame(rows).sort_values("idx").drop(columns=["idx"])
    out_df["model"] = df["model_name"].astype(str).values
    dest = ROOT / "data" / "training"
    dest.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(dest / "spans_coai_train.parquet", index=False)
    print(f"wrote {dest / 'spans_coai_train.parquet'}", flush=True)
    print(f"pile: {out_df['pile'].value_counts().to_dict()}", flush=True)
    print("docs with spans:", int(out_df["spans"].map(lambda s: s != "[]").sum()), "/", len(out_df), flush=True)


if __name__ == "__main__":
    main()
