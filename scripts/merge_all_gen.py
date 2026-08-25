#!/usr/bin/env python3
"""Merge ALL generated data (fictpair / rewrites / premise-fresh) into v2 train.

Pair-key contract (slopdet.pairs joins human+AI on identical split_hint):
  - rewrite rows (paraphrase_seeds):  hint 'para:<prompt_id>'  -> stamped onto the
    matched human row too (match by normalized text prefix)
  - respond rows from premises_ready (gutenberg/blog premises):
    hint 'premise:<prompt_id>' -> stamped onto matched human chunk
  - respond rows from wp prompts (fictpair):
    hint 'fictpair:<wp_prompt_id>' -> stamped onto matched wp human story

Usage:
  uv run python scripts/merge_all_gen.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_COLS = ["text", "label", "register", "spans",
            "source_dataset", "generator", "generation_method",
            "decoding", "split_hint"]

# real model behind each provider during this generation campaign
# (records carry rec['provider']; models were pinned in oracle ~/slopgen/.env)
GENERATOR_BY_PROVIDER = {
    "opencode": "ox-alpha-free",          # opencode.ai/zen/go/v1
    "openrouter": "gemma-4-26b-a4b-it",   # pre-switch free tier
    "tokenrouter": "deepseek-v4-pro-0813",
    "gemini": "gemini-2.5-flash",
    "commandcode": "laguna-s-2.1",
}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def prefix(t: str, n: int = 150) -> str:
    return norm(t)[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=ROOT / "data/v2/v2_train.parquet.labeled.parquet")
    ap.add_argument("--gen-dir", type=Path, default=ROOT / "artifacts/wp_gen")
    ap.add_argument("--seeds", type=Path, default=ROOT / "data/wp/paraphrase_seeds.parquet")
    ap.add_argument("--premises", dest="premises", type=Path,
                    default=ROOT / "data/wp/premises.parquet")
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--max-per-key", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    # index human rows by normalized prefix for stamping
    human_prefix: dict[str, int] = {}
    for i, t in zip(df.index, df["text"]):
        human_prefix.setdefault(prefix(t), i)

    new_rows: list[dict] = []
    stamps: dict[int, str] = {}  # train index -> new hint
    stats: Counter = Counter()

    def add(text: str, method: str, key_hint: str, decoding: str,
            src: str, provider: str = "opencode"):
        t = norm(text)[:args.max_chars]
        if len(t) < args.min_chars:
            stats[f"{src}:too_short"] += 1
            return
        n_key = per_key.get(key_hint, 0)
        if n_key >= args.max_per_key:
            stats[f"{src}:cap"] += 1
            return
        per_key[key_hint] = n_key + 1
        new_rows.append({
            "text": t, "label": 1, "register": f"{method}_pair", "spans": [],
            "source_dataset": src, "generator": GENERATOR_BY_PROVIDER.get(provider, provider),
            "generation_method": method, "decoding": decoding,
            "split_hint": key_hint,
        })
        stats[src] += 1
        # stamp the paired human row if present in train
        # (caller supplies the human lookup separately)

    # ---- 1. rewrites: stories.jsonl? no — rewrites.jsonl keyed by seeds ids ----
    per_key: dict[str, int] = {}
    seeds = pd.read_parquet(args.seeds)
    seed_story = {r["prompt_id"]: r["story"] for _, r in seeds.iterrows()}
    rw = args.gen_dir / "rewrites.jsonl"
    if rw.exists():
        for line in rw.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec or "text" not in rec:
                continue
            pid = rec["prompt_id"]
            add(rec["text"], rec.get("mode", "rewrite"), f"para:{pid}",
                f"temp={rec.get('temp','')}", f"rewrite:{pid.split(':')[0]}",
                provider=rec.get("provider", "opencode"))
            # stamp human original (its full text is in seeds['story'])
            story = seed_story.get(pid)
            if story:
                idx = human_prefix.get(prefix(story))
                if idx is not None:
                    stamps[idx] = f"para:{pid}"

    # ---- 2. fresh from premises (gutenberg/blog) ----
    prem = args.gen_dir / "stories.jsonl"
    pre_file = ROOT / "data/wp/premises_ready.parquet"
    prem_map = {}
    if pre_file.exists():
        pr = pd.read_parquet(pre_file)
        prem_map = {r["prompt_id"]: r["prompt"] for _, r in pr.iterrows()}
    # legacy rows may carry the premise TEXT as prompt_id -> map back to seed id
    premise_jsonl = ROOT / "data/wp/premises.parquet"
    premise_to_seed = {}
    if premise_jsonl.exists():
        for line in premise_jsonl.read_text().splitlines():
            try:
                rec = json.loads(line)
                if "premise" in rec:
                    premise_to_seed[norm(rec["premise"])[:150]] = rec.get("seed_id", "")
            except json.JSONDecodeError:
                continue
    if prem.exists() and prem_map:
        for line in prem.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec or "text" not in rec:
                continue
            pid = rec["prompt_id"]
            if pid not in seed_story:
                # legacy: prompt_id IS the premise text — recover seed id
                sid = premise_to_seed.get(norm(pid)[:150])
                if sid:
                    pid = sid
            add(rec["text"], rec.get("mode", "respond"), f"premise:{pid}",
                f"temp={rec.get('temp','')}", f"premise:{str(pid).split(':')[0]}",
                provider=rec.get("provider", "opencode"))
            story = seed_story.get(pid) or ""
            if not story and pid in prem_map:
                pass  # premise-only seed: no human chunk to pair
            if story:
                idx = human_prefix.get(prefix(story))
                if idx is not None:
                    stamps[idx] = f"premise:{pid}"

    # ---- 3. fictpair fresh from real WP prompts ----
    fictpair = args.gen_dir / "fictpair_stories.jsonl"
    if fictpair.exists():
        for line in fictpair.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec or "text" not in rec:
                continue
            add(rec["text"], "direct", f"fictpair:{rec['prompt_id']}",
                f"temp={rec.get('temp','')}", "fictpair",
                provider=rec.get("provider", "opencode"))

    out = df.copy()
    for idx, hint in stamps.items():
        if idx in out.index:
            out.at[idx, "split_hint"] = hint
    add_df = pd.DataFrame(new_rows)[OUT_COLS]
    out = pd.concat([out, add_df], ignore_index=True)

    n_paired_hints = int(out.split_hint.astype(str).str.startswith(
        ("para:", "premise:", "fictpair:")).sum())
    print(f"[mergegen] added {len(add_df)} AI rows; stamped {len(stamps)} humans")
    print(f"[mergegen] stats: {dict(stats)}")
    print(f"[mergegen] pairable hint rows total: {n_paired_hints}")
    if args.dry_run:
        print("[mergegen] dry-run, nothing written")
        return 0

    args.train.replace(args.train.with_suffix(".parquet.bak"))
    tmp = args.train.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(args.train)
    print(f"[mergegen] wrote {args.train}: {len(df)} -> {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
