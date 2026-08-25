#!/usr/bin/env python3
"""Label the v2 corpus with deterministic FULL spans (why/fix joined from
the frozen ontology). Multiprocessed; explain(sentences=False) skips the
per-sentence pass.

Reads data/v2_train.parquet (or --in) and writes <path>.labeled.parquet
atomically (tmp file + rename). With --holdouts, also processes
v2_holdout*.parquet next to it. Every output row is validated with
spans.validate_row; failures are collected and reported, not written.

Usage:
  uv run python scripts/label_v2_batch.py
  uv run python scripts/label_v2_batch.py --in data/v2_train.parquet --limit 500
  uv run python scripts/label_v2_batch.py --holdouts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from slopdet import spans  # noqa: E402
from slopdet.explain import explain  # noqa: E402

_ONTOLOGY: dict[str, dict[str, str]] | None = None


def load_ontology() -> dict[str, dict[str, str]]:
    """Index ontology/patterns.*.yaml -> {id: {why, fix}}. Loaded lazily per
    worker process (Pool workers re-import this module)."""
    global _ONTOLOGY
    if _ONTOLOGY is not None:
        return _ONTOLOGY
    idx: dict[str, dict[str, str]] = {}
    for p in sorted((ROOT / "ontology").glob("patterns.*.yaml")):
        for entry in yaml.safe_load(p.read_text(encoding="utf-8")) or []:
            if not isinstance(entry, dict) or "id" not in entry:
                continue
            idx[entry["id"]] = {"why": "Rule: " + str(entry.get("fix", "")).strip(),
                                "fix": str(entry.get("fix", "")).strip()}
    _ONTOLOGY = idx
    return idx


def full_spans(out: dict) -> list[dict]:
    """explain() hits -> FULL span dicts (why/fix joined from ontology)."""
    onto = load_ontology()
    spans_out = []
    for hit in (out.get("why_slop") or []) + (out.get("why_human") or []):
        quote = hit.get("quote")
        if not quote:
            continue
        entry = onto.get(hit["id"], {})
        spans_out.append({
            "id": hit["id"],
            "lane": hit.get("lane", "style"),
            "lean": hit.get("lean", "slop"),
            "start": int(hit.get("start", 0)),
            "end": int(hit.get("end", 0)),
            "quote": quote,
            "why": entry.get("why", ""),
            "fix": entry.get("fix", ""),
        })
    return spans_out


def label_row(job: tuple[int, str]) -> tuple[int, list[dict] | None, str | None]:
    """Returns (idx, spans, error). error set when explain failed."""
    idx, text = job
    try:
        out = explain(text, sentences=False)
        return idx, full_spans(out), None
    except Exception as e:  # noqa: BLE001 — one bad doc must not kill the run
        return idx, None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "data" / "v2_train.parquet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--holdouts", action="store_true",
                    help="also process v2_holdout*.parquet next to --in")
    args = ap.parse_args()

    paths = [args.inp]
    if args.holdouts:
        paths += sorted(args.inp.parent.glob("v2_holdout*.parquet"))
        paths = [p for p in paths if not p.name.endswith(".labeled.parquet")]

    for path in paths:
        df = pd.read_parquet(path)
        jobs = [(i, str(t)) for i, t in enumerate(df["text"])]
        if args.limit:
            jobs = jobs[: args.limit]
        print(f"[v2label] {path.name}: {len(jobs)} docs, workers={args.workers}", flush=True)
        t0 = time.time()
        results: dict[int, tuple[list[dict] | None, str | None]] = {}
        with Pool(args.workers) as pool:
            for idx, spans_out, err in pool.imap_unordered(label_row, jobs, chunksize=16):
                results[idx] = (spans_out, err)
        print(f"[v2label] labeled {len(results)} in {time.time() - t0:.0f}s", flush=True)

        ok = bad = 0
        lanes: dict[str, int] = {}
        errors: list[str] = []
        new_spans: list = [None] * len(jobs)
        for idx, (spans_out, err) in results.items():
            row = {"text": str(df["text"].iloc[idx]), "label": int(df["label"].iloc[idx]),
                   "register": str(df["register"].iloc[idx]), "spans": spans_out or []}
            if err is not None:
                bad += 1
                errors.append(f"row {idx}: {err}")
                new_spans[idx] = []
                continue
            try:
                spans.validate_row(row)
                ok += 1
                new_spans[idx] = spans_out
                for s in spans_out:
                    if s["lean"] == "slop":
                        lanes[s["lane"]] = lanes.get(s["lane"], 0) + 1
            except ValueError as e:
                bad += 1
                errors.append(f"row {idx}: {e}")
                new_spans[idx] = []
        df = df.copy()
        df["spans"] = new_spans
        tmp = path.with_suffix(path.suffix + ".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.rename(path.with_suffix(path.suffix + ".labeled.parquet"))
        print(f"[v2label] ok={ok} bad={bad} lanes_fired={lanes}", flush=True)
        for e in errors[:10]:
            print(f"[v2label]   {e}", flush=True)
        print(f"[v2label] wrote {path.with_suffix(path.suffix + '.labeled.parquet')}", flush=True)


if __name__ == "__main__":
    main()
