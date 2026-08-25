#!/usr/bin/env python3
"""Distill gutenberg/blog chunks into one-line premises for fresh AI generation."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_ai_stories import load_env_key  # reuse key loading

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "data/wp/paraphrase_seeds.parquet")
    ap.add_argument("--registers", default="gutenberg,blogs")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data/wp/premises.parquet")
    args = ap.parse_args()

    key = load_env_key()
    model = os.environ.get("OPENROUTER_MODEL", "stealth/ox-alpha")
    import requests
    df = pd.read_parquet(args.inp)
    reg_filter = set(args.registers.split(","))
    # prompt col holds '[source register: X]' tag
    df = df[df.prompt.str.extract(r"\[source register: (\w+)\]")[0].isin(reg_filter)]
    rows = df.to_dict("records")
    print(f"[premise] {len(rows)} chunks to distill", flush=True)

    out_path = args.out
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try: done.add(json.loads(line)["seed_id"])
            except Exception: pass
    print(f"[premise] {len(done)} already done", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    def work(rec):
        sid, text = rec["prompt_id"], rec["story"]
        payload = {"model": model, "temperature": 0.3, "messages": [
            {"role": "system", "content": "Summarize story premises. Output ONLY the premise, one sentence, no commentary."},
            {"role": "user", "content": f"Distill this passage into a one-sentence story premise it could have been written from:\n\n{text[:3000]}"}]}
        for a in range(5):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                    json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=90)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(min(60, 3*2**a)); continue
                r.raise_for_status()
                premise = r.json()["choices"][0]["message"]["content"].strip()[:300]
                return {"seed_id": sid, "premise": premise}
            except Exception:
                time.sleep(min(60, 3*2**a))
        return {"seed_id": sid, "error": "failed"}

    todo = [r for r in rows if r["prompt_id"] not in done]
    n_ok = 0
    with out_path.open("a") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, r) for r in todo[:args.limit]]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                if "error" not in rec: n_ok += 1
                if i % 50 == 0:
                    print(f"[premise] {i}/{len(todo)} ok={n_ok}", flush=True)
    print(f"[premise] done: {n_ok}/{min(len(todo), args.limit)}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
