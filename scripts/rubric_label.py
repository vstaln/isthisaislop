#!/usr/bin/env python3
"""Concurrent LLM-judge labeling engine for the v2 slop corpus.

Reads a parquet (--in), builds a deterministic doc_id per row
(f"{row_index}:{sha1(text)[:12]}"), and asks a judge model for a JSON
rubric verdict defined by the taxonomy system prompt
(research/taxonomy_v2_system_prompt.txt -> {"ai_verdict", "confidence",
"scores", "spans"}). With --batch-docs N > 1, up to N docs share one API
call: they are numbered DOC 1..N in the user message and the system prompt
is extended to demand {"results": [...one object per doc, same order...]}.
Output is still ONE JSON line PER DOC (not per batch) appended to
artifacts/v2_rubric/<stem>.jsonl; docs already present by doc_id are never
re-labeled (resume). If a batch response's array is shorter than the batch,
missing docs are marked {"error": "missing_in_batch"} and retried
individually once. Docs whose call exhausts retries get {"error": ...}
lines and do not block the queue. End of run prints + writes a summary of
register x ai_verdict counts vs provenance agreement.

Concurrency: ThreadPoolExecutor over requests (--workers thread pools of
batches, --workers default 8). Retries: up to 6 per API call, backoff
min(90, 3*2^attempt)s on 429/5xx/timeouts.

Any OpenAI-compatible /chat/completions endpoint works — OpenRouter is just the
default base URL, not a requirement. Env (the OPENROUTER_* spellings are still
read as fallbacks so older invocations keep working):

  ITAIS_JUDGE_BASE     base URL, default https://openrouter.ai/api/v1
  ITAIS_JUDGE_MODEL    model id, default stealth/ox-alpha
  ITAIS_JUDGE_API_KEY  bearer token, also parsed from repo .env (no dotenv import)

Usage:
  uv run python scripts/rubric_label.py --in data/v2_train.parquet
  uv run python scripts/rubric_label.py --in data/v2_train.parquet --limit 200 --shuffle
  uv run python scripts/rubric_label.py --in data/v2_train.parquet --batch-docs 4 --workers 8
  uv run python scripts/rubric_label.py --in data/v2_train.parquet --dry-run   # no network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# HTTP backend: prefer requests; fall back to httpx (installed transitively).
try:
    import requests  # type: ignore

    _POST = requests.post
    _TIMEOUT_ERRS = (requests.Timeout, requests.ConnectionError)
    _HTTP_ERRS = (requests.HTTPError,)
except ImportError:  # pragma: no cover - exercised on envs without requests
    import httpx  # type: ignore

    _POST = httpx.post
    _TIMEOUT_ERRS = (httpx.TimeoutException, httpx.TransportError)
    _HTTP_ERRS = (httpx.HTTPStatusError, httpx.HTTPError)

ROOT = Path(__file__).resolve().parents[1]
RUBRIC_DIR = ROOT / "artifacts" / "v2_rubric"

DEFAULT_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "stealth/ox-alpha"


def judge_env(name: str, default: str = "") -> str:
    """ITAIS_JUDGE_<name>, falling back to the older OPENROUTER_<name> spelling."""
    return (os.environ.get(f"ITAIS_JUDGE_{name}")
            or os.environ.get(f"OPENROUTER_{name}")
            or default)


API_URL = judge_env("BASE", DEFAULT_BASE).rstrip("/") + "/chat/completions"
MAX_RETRIES = 6
TEXT_TRUNC = 6000

BATCH_SUFFIX = (
    "\n\nBATCHING MODE: The user message contains multiple documents, "
    "numbered DOC 1..N. Judge EACH document INDEPENDENTLY against the "
    "rubric above. Respond with ONE JSON object of exactly this shape:\n"
    '{"results": [ ...one output object per document, in the SAME order '
    'as DOC 1..N... ]}\nThe results array MUST contain exactly one entry '
    "per document."
)


def load_env_key() -> str:
    """The judge API key from the environment, else parsed out of the repo .env."""
    key = judge_env("API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("ITAIS_JUDGE_API_KEY", "OPENROUTER_API_KEY"):
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


def doc_header(register: str, generator: str) -> str:
    gen = f"{generator}"
    return (f"REGISTER: {register}\n"
            f"GENERATOR-METADATA (do NOT peek for the verdict, judge the text): {gen}")


def build_prompt(register: str, generator: str, text: str) -> str:
    """User prompt for a single document."""
    return f"{doc_header(register, generator)}\n\nTEXT:\n{text[:TEXT_TRUNC]}"


def build_batch_prompt(docs: list[dict]) -> str:
    """User prompt for a multi-document batch: DOC 1..N blocks."""
    blocks = [
        f"DOC {j}\n{doc_header(d['register'], str(d.get('generator', '')))}"
        f"\n\nTEXT:\n{str(d['text'])[:TEXT_TRUNC]}"
        for j, d in enumerate(docs, start=1)
    ]
    return "\n\n".join(blocks)


def call_judge(model: str, key: str, system: str, user: str,
                    expect_key: str = "ai_verdict",  # normalized via _norm_result
                    timeout: int = 180) -> tuple[dict, int]:
    """One API call. Returns (parsed_json, raw_response_chars).

    Up to MAX_RETRIES attempts, backoff min(90, 3*2^attempt) on
    429/5xx/timeouts/bad bodies. Raises RuntimeError when exhausted.
    """
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            r = _POST(API_URL, json=payload, headers=headers, timeout=timeout)
            raw_chars = len(r.text)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(90, 3 * 2 ** attempt))
                continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                # some frontier models emit reasoning-only or filtered responses
                content = msg.get("reasoning") or ""
            if not content:
                raise ValueError("empty content (no message.content)")
            result = parse_json(content)
            result = _norm_result(result) if isinstance(result, dict) else result
            if not isinstance(result, dict) or expect_key not in result:
                raise ValueError(f"missing {expect_key}: keys={sorted(result) if isinstance(result, dict) else type(result)}")
            return result, raw_chars
        except _TIMEOUT_ERRS:
            last_err = "timeout"
            time.sleep(min(90, 3 * 2 ** attempt))
        except _HTTP_ERRS as e:
            last_err = str(e)
            time.sleep(min(90, 3 * 2 ** attempt))
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            last_err = f"bad body: {e}"
            time.sleep(min(90, 3 * 2 ** attempt))
    raise RuntimeError(f"judge endpoint failed after {MAX_RETRIES} attempts: {last_err}")


def done_doc_ids(shard: Path) -> set[str]:
    """Resume: scan existing jsonl for doc_id keys."""
    done: set[str] = set()
    if shard.exists():
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["doc_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def _base_rec(row: dict) -> dict:
    return {
        "doc_id": row["doc_id"],
        "register": row["register"],
        "label": row["label"],
        "generator": row.get("generator", ""),
    }


def label_single(model: str, key: str, system: str, row: dict) -> dict:
    """Label one doc via its own API call; returns a jsonl record (never raises)."""
    try:
        result, raw_chars = call_judge(
            model, key, system,
            build_prompt(row["register"], str(row.get("generator", "")), str(row["text"])),
        )
    except RuntimeError as e:
        return {**_base_rec(row), "error": str(e), "raw_response_chars": 0}
    return {**_base_rec(row), "result": result, "raw_response_chars": raw_chars}


def _norm_result(d: dict) -> dict:
    """Accept both 'verdict' (taxonomy_v2) and 'ai_verdict' (legacy) keys."""
    if "ai_verdict" not in d and "verdict" in d:
        d = {**d, "ai_verdict": d["verdict"]}
    return d


def label_batch(model: str, key: str, system: str,
                batch_sys: str, docs: list[dict]) -> list[dict]:
    """Label a batch of docs in one API call.

    On success distributes result["results"] positionally; docs missing from
    the array (or malformed entries) get error "missing_in_batch" and are
    retried individually once. If the batch call itself exhausts retries,
    every doc gets the batch error. Never raises.
    """
    bases = [_base_rec(d) for d in docs]
    try:
        parsed, raw_chars = call_judge(
            model, key, batch_sys, build_batch_prompt(docs), expect_key="results")
    except RuntimeError as e:
        return [{**b, "error": f"batch: {e}", "raw_response_chars": 0} for b in bases]

    arr = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(arr, list):
        arr = []
    out: dict[int, dict] = {}
    missing: list[int] = []
    for j, b in enumerate(bases):
        if j < len(arr) and isinstance(arr[j], dict) and "ai_verdict" in _norm_result(arr[j]):
            out[j] = {**b, "result": _norm_result(arr[j]), "raw_response_chars": raw_chars}
        else:
            missing.append(j)

    # Defensive: retry any doc the batch dropped, individually, once.
    for j in missing:
        try:
            result, raw_chars = call_judge(
                model, key, system,
                build_prompt(docs[j]["register"], str(docs[j].get("generator", "")),
                             str(docs[j]["text"])),
            )
            out[j] = {**bases[j], "result": result,
                      "raw_response_chars": raw_chars, "retried_individually": True}
        except RuntimeError as e:
            out[j] = {**bases[j],
                      "error": f"missing_in_batch (individual retry failed: {e})",
                      "raw_response_chars": 0}
    return [out[j] for j in range(len(bases))]


def summarize(records: list[dict]) -> dict:
    """Counts by register x ai_verdict vs provenance agreement."""
    reg_x_verdict: Counter[tuple[str, str]] = Counter()
    conf: Counter[tuple[int, str]] = Counter()
    per_reg: dict[str, dict] = defaultdict(lambda: {"n": 0, "agree": 0})
    total = agree = errors = 0
    for rec in records:
        reg = rec.get("register", "?")
        verdict = ("ERROR" if rec.get("error")
                   else str(rec.get("result", {}).get("ai_verdict", "unparsed")))
        reg_x_verdict[(reg, verdict)] += 1
        lab = rec.get("label")
        per_reg[reg]["n"] += 1
        if verdict in ("human", "ai"):
            total += 1
            conf[(lab, verdict)] += 1
            if verdict == ("human" if lab == 0 else "ai"):
                agree += 1
                per_reg[reg]["agree"] += 1
        elif verdict == "ERROR":
            errors += 1
    return {
        "n_records": len(records),
        "n_scored": total,
        "n_errors": errors,
        "agreement_rate": round(agree / total, 4) if total else 0.0,
        "confusion_counts": {
            f"provenance={k[0]}|llm={k[1]}": v for k, v in sorted(conf.items())},
        "per_register": {
            reg: {**d, "agreement_rate": round(d["agree"] / d["n"], 4) if d["n"] else 0.0}
            for reg, d in sorted(per_reg.items())},
        "register_x_verdict": {
            f"{k[0]}|{k[1]}": v for k, v in sorted(reg_x_verdict.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="input parquet")
    ap.add_argument("--taxonomy", type=Path,
                    default=ROOT / "research" / "taxonomy_v2_system_prompt.txt")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    ap.add_argument("--batch-docs", type=int, default=1,
                    help="docs per API call (default 1); >1 enables batching")
    ap.add_argument("--limit", type=int, default=None, help="label at most N docs")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle before slicing so a partial run samples all registers")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate setup (prompt file, resume logic), print first prompt, exit")
    args = ap.parse_args()
    if args.batch_docs < 1:
        ap.error("--batch-docs must be >= 1")

    system = args.taxonomy.read_text(encoding="utf-8")
    batch_sys = system + BATCH_SUFFIX if args.batch_docs > 1 else system
    print(f"[rubric] taxonomy prompt loaded: {args.taxonomy.name} ({len(system)} chars); "
          f"batch={args.batch_docs}", flush=True)

    df = pd.read_parquet(args.inp)
    rows: list[dict] = []
    for i, row in enumerate(df.to_dict("records")):
        text = str(row["text"])
        rows.append({
            "doc_id": f"{i}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}",
            "register": str(row.get("register", "")),
            "label": int(row["label"]) if row.get("label") in (0, 1) else -1,
            "generator": str(row.get("generator", "")),
            "text": text,
        })
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[:args.limit]

    RUBRIC_DIR.mkdir(parents=True, exist_ok=True)
    shard = RUBRIC_DIR / f"{args.inp.stem}.jsonl"
    done = done_doc_ids(shard)
    todo = [r for r in rows if r["doc_id"] not in done]
    n_batches = (len(todo) + args.batch_docs - 1) // args.batch_docs
    print(f"[rubric] {len(rows)} selected ({len(done)} already done in shard), "
          f"{len(todo)} to go in ~{n_batches} batches; workers={args.workers}",
          flush=True)

    if args.dry_run:
        # No network: prove prompt construction + resume-set loading work.
        if todo:
            first = todo[:min(args.batch_docs, len(todo))]
            print("[rubric] --- first user prompt ---", flush=True)
            prompt = (build_batch_prompt(first) if len(first) > 1
                      else build_prompt(first[0]["register"], first[0]["generator"],
                                        first[0]["text"]))
            print(prompt[:1200], flush=True)
        print(f"[rubric] dry-run OK: resume set size={len(done)}", flush=True)
        return 0

    model = judge_env("MODEL", DEFAULT_MODEL)
    key = load_env_key()
    if not key:
        print("[rubric] no ITAIS_JUDGE_API_KEY (env or .env); cannot run", flush=True)
        return 1
    print(f"[rubric] model={model} endpoint={API_URL}", flush=True)

    def run_unit(unit: list[dict]) -> list[dict]:
        if len(unit) == 1:
            return [label_single(model, key, system, unit[0])]
        return label_batch(model, key, system, batch_sys, unit)

    units = [todo[i:i + args.batch_docs] for i in range(0, len(todo), args.batch_docs)]
    n_docs = n_fail = n_batches_done = 0
    t0 = time.time()
    with shard.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_unit, u) for u in units]
            for fut in as_completed(futures):
                n_batches_done += 1
                for rec in fut.result():
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if "error" in rec:
                        n_fail += 1
                        print(f"[rubric] doc {rec['doc_id']} failed: {rec['error']}",
                              flush=True)
                    else:
                        n_docs += 1
                total_now = n_docs + n_fail
                if total_now % 50 == 0 or n_batches_done == len(units):
                    dt = max(time.time() - t0, 1e-9)
                    bps = n_batches_done / dt
                    dps = total_now / dt
                    eta_s = (len(todo) - total_now) / max(dps, 1e-9)
                    eta = f"{int(eta_s // 3600):02d}:{int((eta_s % 3600) // 60):02d}"
                    print(f"[rubric] done={n_docs} fail={n_fail} "
                          f"rate/min={dps * 60:.1f} ({bps:.2f} batches/s, "
                          f"{dps:.2f} docs/s) eta={eta}", flush=True)

    records = []
    for line in shard.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    summary = summarize(records)
    summary_path = RUBRIC_DIR / f"{args.inp.stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[rubric] summary: agreement={summary['agreement_rate']:.3f} "
          f"scored={summary['n_scored']} errors={summary['n_errors']} -> {summary_path}",
          flush=True)
    for reg, val in sorted(summary["per_register"].items()):
        print(f"[rubric]   {reg}: {val}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
