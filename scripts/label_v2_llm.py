#!/usr/bin/env python3
"""OpenRouter rubric-audit of the labeled v2 corpus.

Samples stratified docs (register x label) from the labeled train parquet,
asks an OpenRouter model for a JSON rubric verdict + quoted spans, appends
one JSON line per doc to artifacts/v2_audit/audit_<shard>.jsonl (resumable:
docs already present by doc_idx are skipped), then writes agreement.json
comparing LLM verdicts against provenance labels.

Env: OPENROUTER_MODEL (default stealth/ox-alpha), OPENROUTER_API_KEY
(also read from repo .env — simple parser, no dotenv import).

Usage:
  uv run python scripts/label_v2_llm.py --n 300
  uv run python scripts/label_v2_llm.py --labeled data/v2_train.parquet.labeled.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "artifacts" / "v2_audit"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = """You audit text for machine-generated writing. Judge ONLY the text itself.
Respond with a single JSON object, no prose, matching this schema:
{"verdict":"human|ai|mixed",
 "confidence":0.0-1.0,
 "rubric":{"style_tics":1-5,"structure_uniformity":1-5,"specificity":1-5,
           "cadence_variability":1-5,"cliche_density":1-5},
 "spans":[{"quote":"...","why":"..."}]}
Rules: quote spans VERBATIM from the text, each <=200 chars, max 5 spans.
verdict 'mixed' only if the text genuinely contains both human and AI prose."""


def load_env_key() -> str:
    """OPENROUTER_API_KEY from env, else parsed manually from repo .env."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "OPENROUTER_API_KEY":
                return v.strip().strip('"').strip("'")
    return ""


def strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def parse_json(s: str) -> dict:
    """Defensive JSON parse: strip code fences, grab first {...} block."""
    try:
        return json.loads(strip_fences(s))
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", strip_fences(s), re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def call_openrouter(model: str, key: str, text: str) -> dict:
    """One audit call, 5 attempts, backoff min(60, 5*2^a) on 429/5xx."""
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},  # dropped if rejected
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[:6000]},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err = ""
    for attempt in range(5):
        try:
            r = requests.post(API_URL, json=payload, headers=headers, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            r.raise_for_status()
            body = r.json()
            content = body["choices"][0]["message"]["content"]
            return parse_json(content)
        except requests.HTTPError as e:
            last_err = str(e)
            time.sleep(min(60, 5 * 2 ** attempt))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            last_err = f"bad body: {e}"
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"openrouter failed after 5 attempts: {last_err}")


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    """~n docs spread evenly across (register, label) strata."""
    rng = random.Random(seed)
    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (reg, lab) in enumerate(zip(df["register"], df["label"])):
        strata[(str(reg), int(lab))].append(i)
    per = max(1, n // max(1, len(strata)))
    picked: list[int] = []
    for idxs in sorted(strata):
        k = min(per, len(strata[idxs]))
        picked += rng.sample(strata[idxs], k)
    rng.shuffle(picked)
    return df.iloc[picked[:n]]


def done_indices(shard: Path) -> set[int]:
    """Resume: scan existing jsonl for doc_idx keys."""
    done: set[int] = set()
    if shard.exists():
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(int(json.loads(line)["doc_idx"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def write_agreement(records: list[dict]) -> None:
    """LLM verdict vs provenance label: overall agreement, confusion counts,
    per-register breakdown."""
    v2l = {"human": 0, "ai": 1}
    conf: Counter[tuple[int, str]] = Counter()
    per_reg: dict[str, dict] = defaultdict(lambda: {"n": 0, "agree": 0})
    total = agree = 0
    for rec in records:
        verdict = rec.get("verdict", "")
        lab = rec.get("label")
        reg = rec.get("register", "?")
        per_reg[reg]["n"] += 1
        if verdict in v2l:
            total += 1
            conf[(int(lab), verdict)] += 1
            if v2l[verdict] == int(lab):
                agree += 1
                per_reg[reg]["agree"] += 1
        else:  # 'mixed' or unparsable: counted, but not agreement
            conf[(int(lab), verdict or "unparsed")] += 1
    out = {
        "n_scored": total,
        "agreement_rate": round(agree / total, 4) if total else 0.0,
        "confusion_counts": {f"provenance={k[0]}|llm={k[1]}": v for k, v in sorted(conf.items())},
        "per_register": {reg: {**d, "agreement_rate": round(d["agree"] / d["n"], 4) if d["n"] else 0.0}
                         for reg, d in sorted(per_reg.items())},
    }
    dest = AUDIT_DIR / "agreement.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[v2llm] agreement {out['agreement_rate']:.3f} over {out['n_scored']} -> {dest}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled", type=Path, default=ROOT / "data" / "v2_train.parquet.labeled.parquet")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shard", type=Path, default=None, help="jsonl shard (default artifacts/v2_audit/audit_0.jsonl)")
    args = ap.parse_args()

    model = os.environ.get("OPENROUTER_MODEL", "stealth/ox-alpha")
    key = load_env_key()
    if not key:
        print("[v2llm] no OPENROUTER_API_KEY (env or .env); cannot run", flush=True)
        return 1

    df = pd.read_parquet(args.labeled)
    sample = stratified_sample(df, args.n)
    print(f"[v2llm] sampled {len(sample)} of {len(df)} docs; model={model}", flush=True)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    shard = args.shard or (AUDIT_DIR / "audit_0.jsonl")
    done = done_indices(shard)
    todo = [(i, row) for i, row in enumerate(sample.to_dict("records")) if i not in done]
    print(f"[v2llm] resume: {len(done)} already done, {len(todo)} to go", flush=True)

    n_new = 0
    with shard.open("a", encoding="utf-8") as f:
        for i, row in todo:
            try:
                result = call_openrouter(model, key, str(row["text"]))
            except RuntimeError as e:
                print(f"[v2llm] doc {i} failed: {e}", flush=True)
                continue
            rec = {"doc_idx": i, "register": row.get("register", ""),
                   "label": int(row["label"]), "generator": row.get("generator", ""),
                   "model": model, **result}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % 25 == 0:
                print(f"[v2llm] {n_new}/{len(todo)} done", flush=True)

    print(f"[v2llm] wrote {n_new} new records -> {shard}", flush=True)
    records = []
    for line in shard.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if records:
        write_agreement(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
