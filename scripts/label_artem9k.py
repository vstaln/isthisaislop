#!/usr/bin/env python3
"""Label the artem9k ai-text-detection-pile (1.39M docs) with deterministic spans.

The pile has doc-level labels (source = human/ai) but no spans. This script runs
explain() on each doc and writes a spans parquet compatible with
build_training_parquet.py (text / label / pile / spans).

Strategy (from the T4-prep discussion):
- AI docs: all (364k, short — median 366 tok, 9.5 docs/s)
- human docs: subsample, shortest-first, to match the 512-token window
  (docs > ~2k chars are mostly truncated away anyway)

Progress: writes part_*.parquet sidecars to /tmp/label_artem9k/ every chunk,
so a background run can be monitored by counting sidecars. Also appends a
human-readable progress line. Safe to kill/resume (parts are idempotent).

Usage:
  uv run python scripts/label_artem9k.py --ai-all --human-n 200000 --workers 4
  # resume after a crash: re-run the same command; done parts are skipped
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

ARTEM9K_DIR = ROOT / "data" / "raw" / "artem9k"
SHARDS = sorted(ARTEM9K_DIR.glob("t*.parquet"))
OUT = ROOT / "data" / "training" / "spans_artem9k_train.parquet"
TMPDIR = Path("/tmp/label_artem9k")


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


def load_corpus(label: int, human_n: int, seed: int = 0) -> pd.DataFrame:
    """Load all docs with `label`, subsampling humans shortest-first."""
    frames = []
    for shard in SHARDS:
        df = pd.read_parquet(shard, columns=["id", "source", "text"])
        keep = df["source"] == ("ai" if label == 1 else "human")
        df = df[keep].copy()
        df["label"] = label
        df["pile"] = label
        frames.append(df[["id", "text", "label", "pile"]])
    out = pd.concat(frames, ignore_index=True)
    if label == 0 and human_n and len(out) > human_n:
        out["_len"] = out["text"].str.len()
        out = out.sort_values("_len").head(human_n).drop(columns="_len")
    print(f"  {label}: {len(out)} docs (source={'ai' if label else 'human'})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-all", action="store_true", help="label all AI docs (default)")
    ap.add_argument("--human-n", type=int, default=200_000, help="human docs to label (shortest-first)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    TMPDIR.mkdir(parents=True, exist_ok=True)
    print(f"resume: parts in {TMPDIR}", flush=True)

    # AI first (all), then human
    corpus = load_corpus(1, human_n=None)
    if args.human_n:
        corpus = pd.concat([corpus, load_corpus(0, args.human_n)], ignore_index=True)

    print(f"total {len(corpus)} docs to label", flush=True)
    if args.dry_run:
        return

    t0 = time.time()
    n = len(corpus)
    done = 0
    for part, start in enumerate(range(0, n, args.chunk)):
        part_path = TMPDIR / f"part_{part:04d}.parquet"
        if part_path.exists():
            print(f"  skip part {part} (already labeled)", flush=True)
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

    # merge parts → final parquet
    parts = sorted(TMPDIR.glob("part_*.parquet"))
    new = pd.concat(pd.read_parquet(p) for p in parts)
    new = new.drop_duplicates(subset="id")
    out = new[["id", "text", "label", "pile", "spans", "slop_tags", "human_tags"]]
    out = out.reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} docs, {(time.time()-t0)/3600:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
