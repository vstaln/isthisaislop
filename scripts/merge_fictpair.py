#!/usr/bin/env python3
"""Merge generated WP fiction pairs into the v2 labeled training parquet.

Reads artifacts/wp_gen/{stories,rewrites}.jsonl (ox-alpha output), joins each
prompt_id to its human story via data/wp/prompts.parquet, emits:
  - AI rows   (label 1): register 'fictpair', method direct/rewrite, hint 'fictpair:<pid>'
  - human rows stay in writingprompts register untouched — pairing happens at
    train time via slopdet.pairs.build_pairs on the shared hint key.

Dedupes against existing rows (exact text + doc_id). Writes
data/v2/v2_train.labeled.parquet atomically after backup.

Usage:
  uv run python scripts/merge_fictpair.py [--min-chars 800] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_COLS = ["text", "label", "register", "spans",
            "source_dataset", "generator", "generation_method",
            "decoding", "split_hint"]


def norm(t: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(t)).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wp-prompts", type=Path, default=ROOT / "data/wp/prompts.parquet")
    ap.add_argument("--train", type=Path, default=ROOT / "data/v2/v2_train.parquet.labeled.parquet")
    ap.add_argument("--gen-dir", type=Path, default=ROOT / "artifacts/wp_gen")
    ap.add_argument("--min-chars", type=int, default=800)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--max-per-prompt", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.wp_prompts.exists():
        raise SystemExit(f"missing {args.wp_prompts} — run scripts/fetch_wp_prompts.py first")

    wp = pd.read_parquet(args.wp_prompts)
    # map prompt_id -> normalized human story (for sanity check only)
    pid_story = {str(r["prompt_id"]): norm(r.get("story", ""))[:120]
                 for _, r in wp.iterrows()}

    ai_rows: list[dict] = []
    seen_texts: set[str] = set()
    stats = {"respond": 0, "rewrite": 0, "too_short": 0, "dup": 0}
    for shard_name, method in [("stories.jsonl", "direct"), ("rewrites.jsonl", "rewrite")]:
        shard = args.gen_dir / shard_name
        if not shard.exists():
            print(f"[merge] {shard_name}: missing, skipped")
            continue
        per_pid: dict[str, int] = {}
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec or "text" not in rec:
                continue
            t = norm(rec["text"])
            if len(t) < args.min_chars:
                stats["too_short"] += 1
                continue
            t = t[:args.max_chars]
            key = hashlib.md5(t.lower().encode()).hexdigest()
            if key in seen_texts:
                stats["dup"] += 1
                continue
            n = per_pid.get(rec["prompt_id"], 0)
            if n >= args.max_per_prompt:
                continue
            per_pid[rec["prompt_id"]] = n + 1
            seen_texts.add(key)
            stats[method] += 1
            ai_rows.append({
                "text": t, "label": 1, "register": "fictpair", "spans": [],
                "source_dataset": "fictpair_gen", "generator": "ox-alpha",
                "generation_method": method, "decoding": f"temp={rec.get('temp', '')}",
                "split_hint": f"fictpair:{rec['prompt_id']}",
            })

    df = pd.read_parquet(args.train)
    before = len(df)
    existing = set(norm(t).lower()[:200] for t in df["text"])
    ai_rows = [r for r in ai_rows if r["text"].lower()[:200] not in existing]
    new_df = pd.DataFrame(ai_rows)[OUT_COLS]
    out = pd.concat([df, new_df], ignore_index=True)

    # pair coverage report
    hints = set(out.loc[out.register == "fictpair", "split_hint"])
    wp_hints = set(out.loc[(out.register == "writingprompts"), "split_hint"]) if \
        "split_hint" in out.columns else set()
    print(f"[merge] ai rows added: {len(ai_rows)} ({stats})")
    print(f"[merge] fictpair prompts: {len(hints)} unique")

    if args.dry_run:
        print("[merge] dry-run, nothing written")
        return 0

    # stamp matching human stories so slopdet.pairs can join on the same key:
    # wp.prompts.parquet maps prompt_id -> the human story; find that story in
    # the writingprompts register and set its hint to fictpair:<prompt_id>
    wp_by_story: dict[str, str] = {}
    for _, r in wp.iterrows():
        s = norm(r.get("story", ""))[:150].lower()
        if len(s) >= 100:
            wp_by_story[s] = f"fictpair:{r['prompt_id']}"
    stamped = 0
    is_wp = out["register"] == "writingprompts"
    for idx in out.index[is_wp]:
        k = norm(out.at[idx, "text"])[:150].lower()
        hint = wp_by_story.get(k)
        if hint:
            out.at[idx, "split_hint"] = hint
            stamped += 1
    print(f"[merge] stamped {stamped} human stories with fictpair hints")

    bak = args.train.with_suffix(".parquet.bak")
    args.train.replace(bak)
    tmp = args.train.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(args.train)
    print(f"[merge] wrote {args.train}: {before} -> {len(out)} rows (backup at {bak.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
