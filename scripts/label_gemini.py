#!/usr/bin/env python3
"""Gemini/Gemma LLM labeling lane — AI vs human lean + checkable spans.

Mirror of scripts/label_laguna.py over the Gemini API (gemma-4-31b-it /
gemma-4-26b-a4b-it). Same SYS prompt, same record schema, same verbatim-quote
guarantee; reuses label_laguna's hydrate/verbatim/doc_id helpers.

Corpus flags (any combination):
  --coai N            N AI + N human docs from data/coai_train.parquet
  --storyscope N      N AI stories from data/raw/storyscope (train split)
  --gutenberg N       N human chunks from data/raw/gutenberg_fiction
  --writingprompts N  N human stories from data/raw/writingprompts
  --scp N             N human stories from data/raw/scp
  --jsonl PATH        generic jsonl with a "text" key (--pile to force target)

Usage:
  uv run python scripts/label_gemini.py --coai 250 --gutenberg 250 \
      --out eval/labels/gemma.jsonl --workers 4

Resumable: ids already carrying a verdict in --out are skipped (a 429/503
mid-batch does not wipe progress). Never writes the API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "scripts"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from label_laguna import SYS, doc_id, hydrate_lanes, sample_coai, verbatim  # noqa: E402

MODEL_DEFAULT = "gemma-4-31b-it"
MODEL_ALT = "gemma-4-26b-a4b-it"
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def load_env() -> dict[str, str]:
    raw: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        raise SystemExit("missing .env with GEMINI_API_KEY")
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        raw[k.strip()] = v.strip()
    if not raw.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY missing from .env")
    return raw


def call_gemini(model: str, api_key: str, text: str, retries: int = 8) -> dict:
    """Label one doc. Returns hydrate_lanes() output + _usage tokens."""
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": SYS + "\n\nTEXT:\n" + text[:6000]}]}
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1200},
    }
    url = API.format(model=model) + "?key=" + api_key
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503):
                time.sleep(min(60, 10 + attempt * 10))
                continue
            time.sleep(5)
            continue
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(min(60, 10 + attempt * 10))
            continue
        try:
            cand = data["candidates"][0]
        except (KeyError, IndexError):
            last = f"no candidates: {str(data)[:200]}"
            time.sleep(10)
            continue
        # Reasoning model: part with thought=True is the chain; answer is the
        # last non-thought part.
        parts = [p for p in cand["content"]["parts"] if not p.get("thought")]
        raw = parts[-1].get("text", "") if parts else ""
        if not raw:
            last = "empty answer"
            time.sleep(5)
            continue
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[-1]
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            last = "no JSON in answer"
            time.sleep(5)
            continue
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            last = "bad JSON"
            time.sleep(5)
            continue
        parsed = hydrate_lanes(parsed)
        # Fabricated quotes die here, same as the laguna lane.
        parsed["style"] = verbatim(text, parsed.get("style") or [])
        parsed["construction"] = verbatim(text, parsed.get("construction") or [])
        parsed["_usage"] = {
            "input": (data.get("usageMetadata") or {}).get("promptTokenCount", 0),
            "output": (data.get("usageMetadata") or {}).get("candidatesTokenCount", 0),
            "thoughts": (data.get("usageMetadata") or {}).get("thoughtsTokenCount", 0),
        }
        return parsed
    raise RuntimeError(f"gemini failed after {retries} tries: {last}")


# ---------------------------------------------------------------- corpora

def sample_storyscope(n: int, seed: int = 1) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(ROOT / "data" / "raw" / "storyscope" / "stories_train.parquet")
    df = df.sample(min(n, len(df)), random_state=seed)
    rows = []
    for rec in df.itertuples(index=False):
        for col in ("story_gpt", "story_claude", "story_gemini", "story_deepseek", "story_kimi"):
            text = str(getattr(rec, col, "") or "").strip()
            if text:
                rows.append(
                    {
                        "id": doc_id(text, "ss-"),
                        "pile": 1,
                        "model": col.replace("story_", ""),
                        "source": "storyscope_train",
                        "text": text,
                    }
                )
    return rows


def _parquet_sample(paths: list[Path], n: int, seed: int) -> "list":
    import pandas as pd

    dfs = [pd.read_parquet(p) for p in paths]
    df = dfs[0] if len(dfs) == 1 else __import__("pandas").concat(dfs, ignore_index=True)
    return df.sample(min(n, len(df)), random_state=seed)


def sample_gutenberg(n: int, seed: int = 1) -> list[dict]:
    df = _parquet_sample(sorted((ROOT / "data" / "raw" / "gutenberg_fiction").glob("train-*.parquet")), n, seed)
    return [
        {
            "id": doc_id(str(r.text), "guten-"),
            "pile": 0,
            "model": "gutenberg",
            "source": "gutenberg_fiction",
            "text": str(r.text),
        }
        for r in df.itertuples(index=False)
    ]


def sample_writingprompts(n: int, seed: int = 1) -> list[dict]:
    df = _parquet_sample(sorted((ROOT / "data" / "raw" / "writingprompts").glob("train-*.parquet")), n, seed)
    return [
        {
            "id": doc_id(str(r.story), "wp-"),
            "pile": 0,
            "model": "writingprompts",
            "source": "writingprompts_train",
            "text": str(r.story),
        }
        for r in df.itertuples(index=False)
    ]


def sample_scp(n: int, seed: int = 1) -> list[dict]:
    paths = sorted((ROOT / "data" / "raw" / "scp").glob("*_cleaned.jsonl"))
    all_rows = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    rng = random.Random(seed)
    picked = rng.sample(all_rows, min(n, len(all_rows)))
    rows = []
    for rec in picked:
        text = rec.get("text") or rec.get("story") or rec.get("content")
        if text:
            rows.append(
                {
                    "id": doc_id(str(text), "scp-"),
                    "pile": 0,
                    "model": "scp",
                    "source": "scp_tales",
                    "text": str(text),
                }
            )
    return rows


def sample_jsonl(path: Path, pile: int | None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = rec.get("text") or rec.get("story") or rec.get("content")
        if text:
            rows.append(
                {
                    "id": rec.get("id") or doc_id(str(text), "jsonl-"),
                    "pile": int(pile) if pile is not None else int(rec.get("pile", -1)),
                    "model": rec.get("model", "jsonl"),
                    "source": path.stem,
                    "text": str(text),
                }
            )
    return rows


# ---------------------------------------------------------------- main

def already_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("gemma"):
            seen.add(rec["id"])
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini/Gemma AI-vs-human labeling lane")
    ap.add_argument("--coai", type=int, default=0)
    ap.add_argument("--storyscope", type=int, default=0)
    ap.add_argument("--gutenberg", type=int, default=0)
    ap.add_argument("--writingprompts", type=int, default=0)
    ap.add_argument("--scp", type=int, default=0)
    ap.add_argument("--jsonl", type=Path)
    ap.add_argument("--pile", type=int, default=None, help="force target for --jsonl")
    ap.add_argument("--limit", type=int, default=0, help="cap total docs (0 = all)")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "labels" / "gemma.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    env = load_env()
    api_key = env["GEMINI_API_KEY"]

    rows: list[dict] = []
    if args.coai:
        rows += sample_coai(args.coai, seed=args.seed)
    if args.storyscope:
        rows += sample_storyscope(args.storyscope, seed=args.seed)
    if args.gutenberg:
        rows += sample_gutenberg(args.gutenberg, seed=args.seed)
    if args.writingprompts:
        rows += sample_writingprompts(args.writingprompts, seed=args.seed)
    if args.scp:
        rows += sample_scp(args.scp, seed=args.seed)
    if args.jsonl:
        rows += sample_jsonl(args.jsonl, args.pile)
    if not rows:
        raise SystemExit("no corpus selected (--coai/--storyscope/--gutenberg/... )")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    done = already_done(args.out)
    todo = [r for r in rows if r["id"] not in done]
    print(f"total {len(rows)} | already done {len(done)} | to label {len(todo)} | model {args.model}", flush=True)
    if not todo:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    usage = {"input": 0, "output": 0, "thoughts": 0}
    failures: list[tuple[str, str]] = []
    done_count = 0
    t0 = time.time()

    def worker(row: dict) -> None:
        nonlocal done_count
        try:
            verdict = call_gemini(args.model, api_key, row["text"])
        except RuntimeError as e:
            with lock:
                failures.append((row["id"], str(e)))
            return
        record = {
            "id": row["id"],
            "pile": row["pile"],
            "model": row["model"],
            "source": row["source"],
            "text": row["text"],
            "gemma": {
                "lean": verdict.get("lean"),
                "style": verdict.get("style") or [],
                "construction": verdict.get("construction") or [],
                "model": args.model,
            },
        }
        with lock:
            with args.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            u = verdict.get("_usage") or {}
            usage["input"] += u.get("input", 0)
            usage["output"] += u.get("output", 0)
            usage["thoughts"] += u.get("thoughts", 0)
            done_count += 1
            if done_count % 25 == 0 or done_count == len(todo):
                rate = done_count / max(time.time() - t0, 1e-9)
                eta = (len(todo) - done_count) / max(rate, 1e-9)
                print(
                    f"  {done_count}/{len(todo)} ({rate:.1f}/s, eta {eta/60:.1f}m) "
                    f"tokens in {usage['input']:,} out {usage['output']:,} thought {usage['thoughts']:,}",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, r) for r in todo]
        for f in as_completed(futs):
            f.result()

    print(
        f"done {done_count}/{len(todo)} | failures {len(failures)} "
        f"| tokens in {usage['input']:,} out {usage['output']:,} thought {usage['thoughts']:,} "
        f"| per doc ≈ {(usage['input']+usage['output']+usage['thoughts'])/max(done_count,1):,.0f}",
        flush=True,
    )
    if failures:
        print("failures:", failures[:10], flush=True)
    print(f"out: {args.out}", flush=True)


if __name__ == "__main__":
    main()
