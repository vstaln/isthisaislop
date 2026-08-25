#!/usr/bin/env python3
"""Build a stratified contamination-audit sample from v2 train.

Takes up to --per-register rows per (register, label) group so every register's
label quality gets measured, not just the big ones. Writes a parquet the
rubric_label.py engine can consume directly.

Usage:
  uv run python scripts/build_audit_sample.py [--per-register 250] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path,
                    default=ROOT / "data/v2/v2_train.parquet.labeled.parquet")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/v2/v2_audit_sample.parquet")
    ap.add_argument("--per-register", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    parts = []
    for _, g in df.groupby(["register", "label"]):
        n = min(len(g), args.per_register)
        parts.append(g.sample(n=n, random_state=args.seed))
    out = pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=args.seed).reset_index(drop=True)

    # doc_id contract used by rubric_label.py: row_index:sha1(text)[:12]
    import hashlib
    ids = [f"{i}:{hashlib.sha1(str(t).encode()).hexdigest()[:12]}"
           for i, t in enumerate(out["text"])]
    out = out.copy()
    out["doc_id_override"] = ids

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    counts = out.groupby(["register", "label"]).size()
    print(f"[audit-sample] wrote {len(out)} docs -> {args.out}")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
