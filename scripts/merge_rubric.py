#!/usr/bin/env python3
"""Merge LLM-judge rubric labels back into the source parquet.

Reads artifacts/v2_rubric/<stem>.jsonl produced by scripts/rubric_label.py,
joins on doc_id (= "<row_index>:<sha1(text)[:12]>"), and adds columns
llm_verdict, llm_confidence, llm_scores (dict), llm_spans (list). The
deterministic spans column is left untouched. Writes <in>.rubric.parquet
atomically (tmp + os.replace) and prints a per-register agreement table.

Usage:
  uv run python scripts/merge_rubric.py --in data/v2_train.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUBRIC_DIR = ROOT / "artifacts" / "v2_rubric"

LLM_COLS = ["llm_verdict", "llm_confidence", "llm_scores", "llm_spans"]


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            out[str(rec["doc_id"])] = rec
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="source parquet")
    args = ap.parse_args()

    shard = RUBRIC_DIR / f"{args.inp.stem}.jsonl"
    if not shard.exists():
        print(f"[merge] no rubric shard at {shard}; run rubric_label.py first", flush=True)
        return 1
    by_id = load_jsonl(shard)
    print(f"[merge] loaded {len(by_id)} rubric records from {shard.name}", flush=True)

    df = pd.read_parquet(args.inp)
    for col in LLM_COLS:
        df[col] = None
    matched = agreed = scored = 0
    per_reg: dict[str, dict] = defaultdict(lambda: {"n": 0, "agree": 0})
    verdict_by_reg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i, row in enumerate(df.to_dict("records")):
        doc_id = f"{i}:{hashlib.sha1(str(row['text']).encode('utf-8')).hexdigest()[:12]}"
        rec = by_id.get(doc_id)
        if rec is None or "result" not in rec:
            continue
        res = rec["result"]
        df.at[i, "llm_verdict"] = str(res.get("ai_verdict", ""))
        conf = res.get("confidence")
        df.at[i, "llm_confidence"] = float(conf) if isinstance(conf, (int, float)) else None
        scores = res.get("scores")
        df.at[i, "llm_scores"] = scores if isinstance(scores, dict) else {}
        spans = res.get("spans")
        df.at[i, "llm_spans"] = spans if isinstance(spans, list) else []
        matched += 1

        reg = str(row.get("register", "?"))
        per_reg[reg]["n"] += 1
        verdict = str(res.get("ai_verdict", ""))
        verdict_by_reg[reg][verdict or "unparsed"] += 1
        lab = row.get("label")
        if verdict in ("human", "ai"):
            scored += 1
            if verdict == ("human" if lab == 0 else "ai"):
                agreed += 1
                per_reg[reg]["agree"] += 1

    dest = args.inp.with_name(args.inp.name + ".rubric.parquet")
    tmp = dest.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, dest)

    overall = round(agreed / scored, 4) if scored else 0.0
    print(f"[merge] matched {matched}/{len(df)} docs; agreement {overall:.3f} "
          f"(scored={scored})", flush=True)
    print(f"[merge] {'register':<24} {'n':>6} {'agree':>7} {'rate':>7}  verdicts", flush=True)
    for reg, d in sorted(per_reg.items()):
        rate = round(d["agree"] / d["n"], 3) if d["n"] else 0.0
        vc = ", ".join(f"{v}={c}" for v, c in sorted(verdict_by_reg[reg].items()))
        print(f"[merge] {reg:<24} {d['n']:>6} {d['agree']:>7} {rate:>7.3f}  {vc}",
              flush=True)
    print(f"[merge] wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
