#!/usr/bin/env python3
"""Fetch WritingPrompts prompts+stories from HF into data/wp/prompts.parquet.

WritingPrompts is the Reddit r/WritingPrompts corpus: a prompt plus human
response stories. We already have the human stories in train_all.parquet
(register='writingprompts') but NOT the prompt texts; this script recovers
(prompt, story) pairs so generate_ai_stories.py can produce AI stories for
the same prompts.

Strategy (no hub datasets lib): try candidate dataset ids IN ORDER against
HF's auto-converted /parquet endpoint
(https://huggingface.co/api/datasets/<id>/parquet -> {config: {split: [urls]}}).
The first id that yields parquet URLs for any config/split wins; all its
shards are downloaded via requests and concatenated. Columns are mapped
defensively by name heuristics after inspecting the actual schema:
  - prompt id:  'prompt_id' | 'id' | first '*id*' column that is not story-side
  - prompt:     'prompt' | 'title' | '*prompt*'
  - story:      'story' | 'response' | 'text' | '*story*'/'*response*'

Output: data/wp/prompts.parquet with columns prompt_id, prompt, story.
Rows where any of the three is missing/empty are dropped; exact duplicate
(prompt_id) rows are deduped keeping the longest story.

Usage:
  uv run python scripts/fetch_wp_prompts.py
  uv run python scripts/fetch_wp_prompts.py --limit 5000 --out data/wp/prompts.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "wp" / "prompts.parquet"
CANDIDATE_IDS = [
    "euclaise/writingprompts",
    "MohamedAshraf701/writing-prompts",
    "GEM/writing_prompts",
]

PROMPT_ID_COLS = ["prompt_id", "id", "promptID", "pid"]
PROMPT_COLS = ["prompt", "title", "source_text"]
STORY_COLS = ["story", "response", "text", "target"]


def _hf_headers() -> dict:
    import os

    tok = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _http_json(url: str):
    import requests

    r = requests.get(url, headers=_hf_headers(), timeout=120)
    r.raise_for_status()
    return r.json()


def _dl_parquet(url: str) -> pd.DataFrame:
    """Download one parquet shard from HF's /parquet endpoint."""
    import io

    import requests
    import pyarrow.parquet as pq

    r = requests.get(url, headers=_hf_headers(), timeout=600)
    r.raise_for_status()
    return pq.read_table(io.BytesIO(r.content)).to_pandas()


def find_dataset() -> tuple[str, list[str]]:
    """First candidate id whose /parquet endpoint returns URLs, plus its URLs."""
    for repo in CANDIDATE_IDS:
        try:
            d = _http_json(f"https://huggingface.co/api/datasets/{repo}/parquet")
        except Exception as e:  # noqa: BLE001
            print(f"[wp] {repo}: no parquet branch ({e})", flush=True)
            continue
        urls = [u for split_urls in d.values()
                if isinstance(split_urls, dict)
                for u in split_urls.values() if u]
        # flatten one more level defensively: some layouts are config->split->list
        flat: list[str] = []
        for item in urls:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, list):
                flat.extend(u for u in item if isinstance(u, str))
        if flat:
            print(f"[wp] using dataset {repo} ({len(flat)} parquet shards)", flush=True)
            return repo, flat
        print(f"[wp] {repo}: parquet endpoint returned no URLs", flush=True)
    raise SystemExit("[wp] no candidate dataset had parquet URLs")


def map_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """Map actual columns to prompt_id/prompt/story by name heuristics."""
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(cands: list[str], exclude: set[str] = frozenset()) -> str | None:
        for cand in cands:
            if cand.lower() in cols and cols[cand.lower()] not in exclude:
                return cols[cand.lower()]
        # substring fallback
        for cand in cands:
            for low, orig in cols.items():
                if cand.lower() in low and orig not in exclude:
                    return orig
        return None

    story_c = pick(STORY_COLS)
    if story_c is None:
        return None
    prompt_c = pick(PROMPT_COLS, exclude={story_c})
    if prompt_c is None:
        return None
    pid_c = pick(PROMPT_ID_COLS, exclude={story_c, prompt_c})
    out = pd.DataFrame({
        "prompt_id": (df[pid_c].astype(str) if pid_c
                      else df[prompt_c].astype(str).str[:80].str.replace(r"\s+", " ", regex=True)),
        "prompt": df[prompt_c].astype(str),
        "story": df[story_c].astype(str),
    })
    out.attrs["mapped"] = {"prompt_id": pid_c or "(derived from prompt)",
                           "prompt": prompt_c, "story": story_c}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="keep at most N pairs")
    args = ap.parse_args()

    repo, urls = find_dataset()
    frames: list[pd.DataFrame] = []
    for i, url in enumerate(urls):
        try:
            part = _dl_parquet(url)
        except Exception as e:  # noqa: BLE001
            print(f"[wp] shard {i} failed ({url.rsplit('/', 1)[-1]}): {e}", flush=True)
            continue
        print(f"[wp] shard {i + 1}/{len(urls)}: {len(part)} rows, "
              f"cols={list(part.columns)}", flush=True)
        frames.append(part)
    if not frames:
        raise SystemExit("[wp] every shard failed")

    raw = pd.concat(frames, ignore_index=True)
    print(f"[wp] downloaded {len(raw)} total rows from {repo}", flush=True)
    df = map_columns(raw)
    if df is None:
        raise SystemExit(f"[wp] could not map columns; saw {list(raw.columns)}")
    print(f"[wp] column mapping: {df.attrs['mapped']}", flush=True)

    before = len(df)
    df = df[(df["prompt"].str.len() > 0) & (df["story"].str.len() > 0)]
    print(f"[wp] dropped {before - len(df)} empty rows -> {len(df)}", flush=True)

    # dedupe on prompt_id, keep longest story per id
    df["_slen"] = df["story"].str.len()
    before = len(df)
    df = (df.sort_values("_slen", ascending=False)
            .drop_duplicates("prompt_id", keep="first")
            .drop(columns="_slen"))
    print(f"[wp] deduped prompt_ids: {before} -> {len(df)}", flush=True)

    if args.limit is not None:
        df = df.head(args.limit)
        print(f"[wp] limited to {args.limit} rows", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["prompt_id", "prompt", "story"]].to_parquet(args.out, index=False)
    n_chars = int(df["story"].str.len().median())
    print(f"[wp] wrote {len(df)} rows -> {args.out} (median story {n_chars} chars)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
