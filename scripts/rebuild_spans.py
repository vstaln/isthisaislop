#!/usr/bin/env python3
"""Rebuild the lean-dependent columns (spans, slop_tags, human_tags) of
existing training parquets using the current ontology/tags.

Reads each spans_*.parquet, re-runs explain(text, sentences=False) on every
doc, and rewrites spans/slop_tags/human_tags in place (pile/text/model kept).
Use after ontology or tag-lean changes so training data matches the detector.
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
sys.path.insert(0, str(ROOT / "scripts"))

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
    idx, text = args
    out = explain(text, sentences=False)
    return {
        "idx": idx,
        "slop_tags": [h["id"] for h in out["why_slop"]],
        "human_tags": [h["id"] for h in out["why_human"]],
        "spans": json.dumps(spans_for(out), ensure_ascii=False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2000, help="docs per write; bounds RAM")
    ap.add_argument("--write", action="store_true", help="actually write; default is dry-run")
    args = ap.parse_args()

    for path in args.paths:
        df = pd.read_parquet(path)
        print(f"{path.name}: {len(df)} docs (chunk {args.chunk})", flush=True)
        t0 = time.time()
        n = len(df)
        tmpdir = Path(f"/tmp/rebuild_{path.stem}")
        tmpdir.mkdir(exist_ok=True)
        for part, start in enumerate(range(0, n, args.chunk)):
            end = min(start + args.chunk, n)
            chunk_df = df.iloc[start:end]
            jobs = [(i, str(t)) for i, t in enumerate(chunk_df["text"])]
            with Pool(args.workers) as pool:
                rebuilt = list(pool.imap_unordered(label_row, jobs, chunksize=32))
            rebuilt.sort(key=lambda r: r["idx"])
            frag = pd.DataFrame(rebuilt).set_index("idx")
            frag.to_parquet(tmpdir / f"part_{part:04d}.parquet")
            print(f"  chunk {start}-{end}/{n} done ({time.time()-t0:.0f}s)", flush=True)
        new = pd.concat(
            pd.read_parquet(p) for p in sorted(tmpdir.glob("part_*.parquet"))
        )
        df["slop_tags"] = new["slop_tags"]
        df["human_tags"] = new["human_tags"]
        df["spans"] = new["spans"]
        if args.write:
            df.to_parquet(path, index=False)
            print(f"  wrote {path.name} in {time.time()-t0:.0f}s", flush=True)
        else:
            # dry-run: report counts
            from collections import Counter

            mis = Counter()
            for s in df["spans"]:
                for sp in json.loads(s):
                    if sp.get("lean") == "slop" and sp["id"] in (
                        "frames", "passive", "weasel",
                    ):
                        mis[sp["id"]] += 1
            print(f"  dry-run: would re-label; remaining mislabeled frames/passive/weasel as slop: {dict(mis) or '0'}", flush=True)


if __name__ == "__main__":
    main()
