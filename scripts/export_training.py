#!/usr/bin/env python3
"""Export eval/labels/*.jsonl into a fine-tuning parquet.

One row per doc: pile target, lean (LLM + local), all three lanes' hits with
verbatim spans, and construction stats as columns. Consumers (span trainer,
scorer) derive features from text; this file carries the labels and spans.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from slopdet.explain import explain  # noqa: E402


def spans_for(rec: dict) -> list[dict]:
    local = rec.get("local") or {}
    out = []
    for hit in (local.get("why_slop") or []) + (local.get("why_human") or []):
        if not hit.get("quote"):
            continue
        out.append(
            {
                "id": hit["id"],
                "lane": hit.get("lane", "style"),
                "lean": hit.get("lean", "slop"),
                "start": hit.get("start"),
                "end": hit.get("end"),
                "quote": hit["quote"],
            }
        )
    return out


def local_tags(rec: dict) -> dict[str, list[str]]:
    local = rec.get("local") or {}
    slop = [h["id"] for h in local.get("why_slop") or []]
    human = [h["id"] for h in local.get("why_human") or []]
    return {"slop_tags": slop, "human_tags": human}


def main() -> None:
    paths = sorted((ROOT / "eval" / "labels").glob("*.jsonl"))
    if not paths:
        raise SystemExit("no label files in eval/labels/")
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    print(f"records {len(rows)} from {[p.name for p in paths]}")

    out = []
    for rec in rows:
        if not rec.get("local"):
            rec["local"] = explain(rec["text"])
        laguna = rec.get("laguna") or rec.get("gemma") or {}
        stats = rec.get("construction_stats") or {}
        tags = local_tags(rec)
        row = {
            "id": rec["id"],
            "pile": int(rec["pile"]),
            "model": rec.get("model"),
            "source": rec.get("source"),
            "labeler": rec.get("labeler"),
            "lean_llm": laguna.get("lean"),
            "lean_local": (rec.get("local") or {}).get("lean"),
            "text": rec["text"],
            "spans": json.dumps(spans_for(rec), ensure_ascii=False),
        }
        row.update(tags)
        row.update({f"stat_{k}": v for k, v in (stats or {}).items()})
        out.append(row)

    df = pd.DataFrame(out)
    dest = ROOT / "data" / "training"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "labeled.parquet", index=False)
    print(f"wrote {len(df)} rows -> {dest / 'labeled.parquet'}")
    print(f"pile: {df['pile'].value_counts().to_dict()}")
    print(f"lean_llm: {df['lean_llm'].value_counts().to_dict()}")
    print(f"lean_local: {df['lean_local'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
