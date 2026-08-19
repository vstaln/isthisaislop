#!/usr/bin/env python3
"""Label the writingprompts reddit human-fiction corpus with deterministic spans.

Same explain() pipeline as label_artem9k.py, but simpler: all docs are human
(label 0). Writes spans_writingprompts_train.parquet for build_training_parquet.

Progress: position-keyed parts in /tmp/label_writingprompts/, resumable.

Usage:
  uv run python scripts/label_writingprompts.py --n 100000 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from slopdet.explain import explain  # noqa: E402

WP_DIR = ROOT / "data" / "raw" / "writingprompts"
OUT = ROOT / "data" / "training" / "spans_writingprompts_train.parquet"
TMPDIR = Path("/tmp/label_writingprompts")


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
    idx, text = args
    out = explain(text, sentences=False)
    return {
        "idx": idx,
        "slop_tags": [h["id"] for h in out["why_slop"]],
        "human_tags": [h["id"] for h in out["why_human"]],
        "spans": json.dumps(spans_for(out), ensure_ascii=False),
    }


def load_corpus(n: int, seed: int = 42) -> pd.DataFrame:
    frames = []
    for shard in sorted(WP_DIR.glob("train-*.parquet")):
        df = pd.read_parquet(shard, columns=["story"])
        df = df.rename(columns={"story": "text"})
        df["id"] = [f"wp_{shard.stem}_{i}" for i in range(len(df))]
        df["label"] = 0
        df["pile"] = 0
        frames.append(df[["id", "text", "label", "pile"]])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="text")
    if n and len(out) > n:
        rng = __import__("random").Random(seed)
        out = out.sample(n, random_state=seed)
    print(f"  {len(out)} docs (human fiction)", flush=True)
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="docs to label")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    TMPDIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.n)
    print(f"total {len(corpus)} docs to label", flush=True)
    if args.dry_run:
        return

    t0 = time.time()
    n = len(corpus)
    done = 0
    for part, start in enumerate(range(0, n, args.chunk)):
        part_path = TMPDIR / f"part_{part:04d}.parquet"
        if part_path.exists():
            done += args.chunk
            continue
        end = min(start + args.chunk, n)
        chunk = corpus.iloc[start:end]
        jobs = [(i, str(t)) for i, t in enumerate(chunk["text"])]
        with Pool(args.workers) as pool:
            labeled = list(pool.imap_unordered(label_row, jobs, chunksize=32))
        labeled.sort(key=lambda r: r["idx"])
        frag = pd.DataFrame(labeled)
        frag["id"] = chunk["id"].values
        frag["label"] = chunk["label"].values
        frag["pile"] = chunk["pile"].values
        frag["text"] = chunk["text"].values
        frag.to_parquet(part_path)
        done += len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        remaining = (n - done) / rate / 3600 if rate else float("nan")
        print(
            f"  chunk {done}/{n} ({done/n:.0%}) ~{rate:.2f} docs/s, "
            f"ETA {remaining:.1f}h, elapsed {elapsed/3600:.1f}h",
            flush=True,
        )

    parts = sorted(TMPDIR.glob("part_*.parquet"))
    new = pd.concat(pd.read_parquet(p) for p in parts)
    new = new.drop_duplicates(subset="id")
    out = new[["id", "text", "label", "pile", "spans", "slop_tags", "human_tags"]]
    out = out.reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} docs, {(time.time()-t0)/3600:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
