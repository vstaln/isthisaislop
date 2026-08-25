#!/usr/bin/env python3
"""Two-stage contamination audit, chained to fire when generation completes.

Stage 0: poll oracle for '[loop] COMPLETE' in ~/slopgen/gen.log
Stage 1: pull final shards, run merge_all_gen.py (labels + pair hints)
Stage 2: stratified sample -> rubric_label.py as BLIND JUDGE (ox-alpha-free)
         disagreements with provenance = suspected contamination
Stage 3: 'why' pass on judged-AI rows with an INDEPENDENT model (gemini-2.5-flash)

Usage:
  nohup uv run python scripts/run_contamination_audit.py > audit.log 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLE = "oracle-old"
GEN_LOG = "/home/ubuntu/slopgen/gen.log"
SHARDS = ["/home/ubuntu/slopgen/artifacts/wp_gen/rewrites.jsonl",
          "/home/ubuntu/slopgen/artifacts/wp_gen/stories.jsonl"]
LOCAL_SHARDS = ROOT / "artifacts/wp_gen"
TRAIN = ROOT / "data/v2/v2_train.parquet.labeled.parquet"
SAMPLE = ROOT / "data/v2/v2_audit_sample.parquet"

JUDGE_ENV = {  # stage 2 judge = ox-alpha-free via opencode-go (same key as chat)
    "ITAIS_JUDGE_BASE": "https://opencode.ai/zen/go/v1",
    "ITAIS_JUDGE_MODEL": "ox-alpha-free",
}
WHY_ENV = {  # stage 3 explainer = independent model family
    "ITAIS_JUDGE_BASE": "https://openrouter.ai/api/v1",
    "ITAIS_JUDGE_MODEL": "google/gemini-2.5-flash",
}


def sh(cmd: str, env_extra: dict | None = None) -> int:
    import os
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"\n[audit] $ {' '.join(cmd[:3])}...", flush=True)
    return subprocess.call(cmd, env=env)


def wait_for_complete() -> None:
    print("[audit] polling oracle for [loop] COMPLETE ...", flush=True)
    while True:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ORACLE,
             f"grep -a '\\[loop\\] COMPLETE' {GEN_LOG} | tail -n 1"],
            capture_output=True, text=True, timeout=60)
        if "COMPLETE" in r.stdout:
            print(f"[audit] generation complete: {r.stdout.strip()}", flush=True)
            return
        time.sleep(600)


def main() -> int:
    wait_for_complete()

    # ---- stage 1: pull + merge ----
    for f in SHARDS:
        subprocess.check_call(["scp", "-q", "-o", "BatchMode=yes",
                               f"{ORACLE}:{f}", str(LOCAL_SHARDS)])
    if sh(["uv", "run", "--with", "pandas", "--with", "pyarrow",
           "python", str(ROOT / "scripts/merge_all_gen.py")]) != 0:
        return 1

    # ---- stage 2: blind judge on stratified sample ----
    if sh(["uv", "run", "--with", "pandas", "--with", "pyarrow",
           "python", str(ROOT / "scripts/build_audit_sample.py"),
           "--per-register", "250"]) != 0:
        return 1
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    env_judge = {"ITAIS_JUDGE_API_KEY": key} if key else {}
    if sh(["uv", "run", "python", str(ROOT / "scripts/rubric_label.py"),
           "--in", str(SAMPLE), "--batch-docs", "4", "--workers", "6"],
          env_extra={**JUDGE_ENV, **env_judge}) != 0:
        return 1

    # ---- stage 3: why pass (independent model) on judged-AI rows ----
    # reuse the same sample; the rubric engine skips docs already labeled,
    # so the why pass runs under a different output dir + judge model.
    why_dir = ROOT / "artifacts/v2_rubric_why"
    why_dir.mkdir(exist_ok=True)
    src = ROOT / "artifacts/v2_rubric/v2_audit_sample.jsonl"
    if src.exists():
        judged_ai = []
        for line in src.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            res = rec.get("result") or {}
            if res.get("verdict") == "ai":
                judged_ai.append(json.dumps(rec, ensure_ascii=False))
        (why_dir / "v2_audit_sample.jsonl").write_text(
            "\n".join(judged_ai) + "\n")
        print(f"[audit] stage 3: {len(judged_ai)} judged-AI docs queued for WHY pass")

    print("[audit] DONE — see artifacts/v2_rubric/v2_train.summary.json + audit outputs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
