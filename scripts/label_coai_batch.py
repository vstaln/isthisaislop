#!/usr/bin/env python3
"""Label a corpus with deterministic spans (one row per doc).

Columns: pile target, slop/human tag lists, verbatim spans. Multiprocessed;
explain(sentences=False) skips the per-sentence pass.

Corpora:
  coai       data/coai_train.parquet (62k docs, label col)
  storyscope data/raw/storyscope/stories_{split}.parquet (5 AI stories per prompt)
  gutenberg  data/raw/gutenberg_fiction/train-*.parquet (human fiction chunks)
  writingprompts data/raw/writingprompts/train-*.parquet (human short fiction)

Usage:
  uv run python scripts/label_coai_batch.py                     # coai (default)
  uv run python scripts/label_coai_batch.py --corpus storyscope --split train
  uv run python scripts/label_coai_batch.py --corpus gutenberg --limit 5000
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


def corpus_rows(corpus: str, split: str, limit: int, min_pg: int = 0) -> tuple[pd.DataFrame, list[tuple]]:
    if corpus == "coai":
        df = pd.read_parquet(ROOT / "data" / "coai_train.parquet")
        jobs = [(i, str(t), int(l)) for i, (t, l) in enumerate(zip(df["text"], df["label"]))]
        return df, jobs
    if corpus == "storyscope":
        df = pd.read_parquet(ROOT / "data" / "raw" / "storyscope" / f"stories_{split}.parquet")
        cols = [c for c in df.columns if c.startswith("story_")]
        jobs = []
        n = 0
        for rec in df.itertuples(index=False):
            for col in cols:
                text = str(getattr(rec, col) or "").strip()
                if not text:
                    continue
                jobs.append((n, text, 1))
                n += 1
                if limit and n >= limit:
                    return df, jobs
        return df, jobs
    if corpus == "gutenberg":
        paths = sorted((ROOT / "data" / "raw" / "gutenberg_fiction").glob("train-*.parquet"))
        dfs = [pd.read_parquet(p) for p in paths]
        df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
        jobs = [(i, str(t), 0) for i, t in enumerate(df["text"].tolist())]
        if min_pg > 0:
            pg = df["file_id"].str.replace("PG", "", regex=False).astype(int)
            keep = pg >= min_pg
            jobs = [(i, str(t), 0) for i, t in enumerate(df["text"][keep].tolist())]
            print(f"  gutenberg filtered to PG>={min_pg}: {len(jobs)} chunks", flush=True)
        return df, jobs
    if corpus == "writingprompts":
        paths = sorted((ROOT / "data" / "raw" / "writingprompts").glob("train-*.parquet"))
        dfs = [pd.read_parquet(p) for p in paths]
        df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
        jobs = [(i, str(t), 0) for i, t in enumerate(df["story"].tolist())]
        return df, jobs
    if corpus == "scp":
        texts: list[str] = []
        for p in sorted((ROOT / "data" / "raw" / "scp").glob("*_cleaned.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = rec.get("text") or rec.get("story") or rec.get("content")
                if text:
                    texts.append(str(text))
        jobs = [(i, t, 0) for i, t in enumerate(texts)]
        return pd.DataFrame({"text": texts}), jobs
    if corpus == "blogs":
        import xml.etree.ElementTree as ET

        texts = []
        for f in sorted((ROOT / "data" / "raw" / "blogs").rglob("*.xml")):
            try:
                root = ET.parse(f).getroot()
            except Exception:  # noqa: BLE001 - one bad file must not kill the corpus
                continue
            posts = [p.text or "" for p in root.iter("post")]
            text = "\n\n".join(posts).strip()
            if len(text) >= 200:  # skip near-empty blog files
                texts.append(text)
        jobs = [(i, t, 0) for i, t in enumerate(texts)]
        return pd.DataFrame({"text": texts}), jobs
    raise SystemExit(f"unknown corpus: {corpus}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="coai", choices=["coai", "storyscope", "gutenberg", "writingprompts", "scp", "blogs"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="random-sample N docs across all shards (overrides --limit)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gutenberg-min-pg", type=int, default=0, help="gutenberg: keep only PG id >= N (recent uploads)")
    args = ap.parse_args()

    df, jobs = corpus_rows(args.corpus, args.split, args.limit, args.gutenberg_min_pg)
    if args.sample:
        import random

        rng = random.Random(args.seed)
        jobs = rng.sample(jobs, min(args.sample, len(jobs)))
    elif args.limit:
        jobs = jobs[: args.limit]
    print(f"{args.corpus} ({args.split}): {len(jobs)} docs", flush=True)
    t0 = time.time()
    with Pool() as pool:
        rows = list(pool.imap_unordered(label_row, jobs, chunksize=16))
    print(f"labeled {len(rows)} in {time.time() - t0:.0f}s", flush=True)

    out_df = pd.DataFrame(rows).sort_values("idx").drop(columns=["idx"])
    dest = ROOT / "data" / "training"
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"spans_{args.corpus}_{args.split}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}", flush=True)
    print(f"pile: {out_df['pile'].value_counts().to_dict()}", flush=True)
    print("docs with spans:", int(out_df["spans"].map(lambda s: s != "[]").sum()), "/", len(out_df), flush=True)


if __name__ == "__main__":
    main()

