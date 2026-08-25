#!/usr/bin/env python3
"""Generate AI stories for WritingPrompts prompts via OpenRouter (v2 corpus).

Reads data/wp/prompts.parquet (columns prompt_id, prompt, story — built by
fetch_wp_prompts.py) and, for each prompt, generates K stories as DIFFERENT
WRITERS: personas are cycled and temperatures vary by index
(persona 0 -> 0.9, 1 -> 1.0, 2 -> 1.1). Concurrency: ThreadPoolExecutor over
(prompt x persona) units. Retries: up to 6 per call, backoff
min(90, 3*2^attempt) on 429/5xx/timeouts/bad bodies.

Resume: artifacts/wp_gen/stories.jsonl, one line per generated story:
  {"prompt_id", "persona_idx", "temp", "text", "chars"}
done-key is f"{prompt_id}:{persona_idx}" — reruns skip completed pairs.

Length guard: a response under MIN_CHARS (800) gets ONE retry with the user
message suffixed '(longer)'; if still short the pair is recorded as
{"prompt_id", "persona_idx", "error": "too_short"} and the run continues.
Progress prints every 25 finished docs: rate + eta. End of run writes
artifacts/wp_gen/stories.summary.json.

Env: OPENROUTER_MODEL (default stealth/ox-alpha), OPENROUTER_API_KEY
(also parsed manually from repo .env — no dotenv import).

Usage:
  uv run python scripts/generate_ai_stories.py --limit 5000 --shuffle
  uv run python scripts/generate_ai_stories.py --stories-per-prompt 3 --workers 6
  uv run python scripts/generate_ai_stories.py --dry-run   # no network
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

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
IN_PARQUET = ROOT / "data" / "wp" / "prompts.parquet"
OUT_DIR = ROOT / "artifacts" / "wp_gen"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 6
MIN_CHARS = 800

PERSONAS = [
    "literary style, rich interiority, varied sentence rhythm",
    "plain conversational storyteller, casual voice",
    "pulpy genre page-turner, fast pacing",
]
TEMPS = {0: 0.85, 1: 0.95, 2: 1.0}  # ox-alpha rejects temp > 1.0 (400)

SYSTEM = (
    "You are a writer responding to a Reddit WritingPrompts prompt. "
    "Write a complete short story responding to the prompt. "
    "400-900 words. Never mention the prompt or being an AI."
)


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


def done_keys(shard: Path) -> set[str]:
    """Resume: scan existing jsonl for prompt_id:persona_idx keys."""
    done: set[str] = set()
    if shard.exists():
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add(f"{rec['prompt_id']}:{rec['persona_idx']}")
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def load_providers() -> list[dict]:
    """All configured chat-completion providers from env/.env, in priority order."""
    raw = {}
    for line in Path(".env").exists() and Path(".env").read_text().splitlines() or []:
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            raw[k.strip()] = v.strip()
    def env(k): return os.environ.get(k) or raw.get(k) or ""
    out = []
    if env("OPENCODE_API_KEY") or env("OPENCODE_GO_API_KEY"):
        out.append({"name": "opencode", "base": env("OPENCODE_BASE") or "https://opencode.ai/zen/go/v1",
                    "model": env("OPENCODE_MODEL") or "ox-alpha-free",
                    "key": env("OPENCODE_API_KEY") or env("OPENCODE_GO_API_KEY")})
    if env("OPENROUTER_API_KEY"):
        out.append({"name": "openrouter", "base": env("OPENROUTER_BASE") or "https://openrouter.ai/api/v1",
                    "model": env("OPENROUTER_MODEL") or "stealth/ox-alpha", "key": env("OPENROUTER_API_KEY")})
    if env("TOKENROUTER_API_KEY"):
        out.append({"name": "tokenrouter", "base": env("TOKENROUTER_BASE") or "https://api.tokenrouter.com/v1",
                    "model": env("TOKENROUTER_MODEL") or "deepseek/deepseek-v4-pro-0813-free", "key": env("TOKENROUTER_API_KEY")})
    if env("GEMINI_API_KEY"):
        out.append({"name": "gemini", "base": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "model": env("GEMINI_MODEL") or "gemini-2.5-flash", "key": env("GEMINI_API_KEY")})
    if env("COMMAND_CODE_API_KEY"):
        out.append({"name": "commandcode", "base": env("COMMAND_CODE_BASE") or "https://api.commandcode.ai/provider/v1",
                    "model": env("COMMAND_CODE_MODEL") or "poolside/laguna-s-2.1-free", "key": env("COMMAND_CODE_API_KEY")})
    return out


def build_user(prompt_text: str, persona: str, longer_hint: bool = False,
               mode: str = "respond", source_text: str = "") -> str:
    if mode == "rewrite":
        user = (f"{persona}\n\nRewrite the following story in your own words. "
                f"Keep the plot, characters, and structure intact, but change "
                f"the phrasing, sentence rhythms, and word choices throughout. "
                f"Do not add commentary.\n\nSTORY:\n{source_text}")
    else:
        user = f"{persona}\n\nPROMPT:\n{prompt_text}"
    return f"{user} (longer)" if longer_hint else user


def generate_story(provider: dict, prompt_text: str, persona: str,
                   persona_idx: int, temp: float,
                   timeout: int = 300, mode: str = "respond",
                   source_text: str = "") -> dict:
    """One unit of work. Returns a record dict; never raises."""
    def attempt(longer_hint: bool):
        nonlocal provider
        last_err = ""
        for a in range(MAX_RETRIES):
            if a > 0 and isinstance(provider, dict) and provider.get("_all"):
                provider = provider["_all"][a % len(provider["_all"])]  # failover rotates
            # url/headers/payload MUST be rebuilt per attempt: after rotation the
            # model name belongs to the new provider, and reusing the previous
            # provider's url/key sends a foreign model name -> deterministic 400.
            url = provider["base"].rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {provider['key']}", "Content-Type": "application/json"}
            payload = {
                "model": provider["model"],
                "temperature": temp,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",
                     "content": build_user(prompt_text, persona, longer_hint,
                                           mode=mode, source_text=source_text)},
                ],
            }
            try:
                r = _POST(url, json=payload, headers=headers, timeout=timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(min(90, 3 * 2 ** a))
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"].get("content") or ""
                if not content.strip():
                    raise ValueError("empty content")
                return content
            except _TIMEOUT_ERRS:
                last_err = "timeout"
                time.sleep(min(90, 3 * 2 ** a))
            except _HTTP_ERRS as e:
                last_err = str(e)
                time.sleep(min(90, 3 * 2 ** a))
            except (KeyError, ValueError) as e:
                last_err = f"bad body: {e}"
                time.sleep(min(90, 3 * 2 ** a))
        raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_err}")

    base = {"prompt_id": "", "persona_idx": persona_idx}
    try:
        text = attempt(False)
        if len(text) < MIN_CHARS:
            print(f"[wpgen] prompt persona {persona_idx}: short ({len(text)} chars), "
                  f"retrying once with '(longer)'", flush=True)
            text = attempt(True)
        if len(text) < MIN_CHARS:
            return {**base, "error": "too_short"}
        return {"prompt_id": "", "persona_idx": persona_idx,
                "temp": temp, "text": text, "chars": len(text)}
    except RuntimeError as e:
        return {**base, "error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=IN_PARQUET)
    ap.add_argument("--limit", type=int, default=5000, help="at most N prompts")
    ap.add_argument("--stories-per-prompt", type=int, default=3,
                    help="personas per prompt (cycled from PERSONAS)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6, help="concurrent API calls")
    ap.add_argument("--mode", choices=["respond", "rewrite"], default="respond",
                    help="respond = fresh story from prompt; rewrite = AI rewrite "
                         "of the human story (paraphrase pairs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print first request payload, exit; no network")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    rows = df.to_dict("records")
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[:args.limit]

    units = []
    for row in rows[:args.stories_per_prompt and len(rows)]:
        for j in range(args.stories_per_prompt):
            units.append((str(row["prompt_id"]), str(row["prompt"]), j,
                          str(row.get("story", ""))))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard = OUT_DIR / ("rewrites.jsonl" if args.mode == "rewrite" else "stories.jsonl")
    done = done_keys(shard)
    todo = [u for u in units if f"{u[0]}:{u[2]}" not in done]
    print(f"[wpgen] {len(units)} prompt-persona pairs selected "
          f"({len(done)} already done in shard), {len(todo)} to go; "
          f"workers={args.workers}", flush=True)

    providers = load_providers()
    if not providers:
        print("[wpgen] no provider API keys found; cannot run", flush=True)
        return 1
    print(f"[wpgen] providers: {[p_['name'] for p_ in providers]}", flush=True)

    if args.dry_run:
        if todo:
            pid, ptext, j, story = todo[0]
            print("[wpgen] --- first request ---", flush=True)
            print(f"[wpgen] providers={[p_['name'] for p_ in providers]} temperature={TEMPS.get(j, 0.9)} mode={args.mode}", flush=True)
            print(f"[wpgen] system={SYSTEM}", flush=True)
            print(f"[wpgen] user={build_user(ptext, PERSONAS[j % len(PERSONAS)], mode=args.mode, source_text=story)[:1200]}",
                  flush=True)
        print(f"[wpgen] dry-run OK: resume set size={len(done)}", flush=True)
        return 0

    if not providers:
        print("[wpgen] no provider keys (env or .env); cannot run", flush=True)
        return 1

    def run_unit(unit: tuple[str, str, int, str], provider_idx: int = 0) -> dict:
        pid, ptext, j, story = unit
        prov = dict(providers[provider_idx % len(providers)])
        prov["_all"] = providers
        rec = generate_story(prov, ptext, PERSONAS[j % len(PERSONAS)],
                             j, TEMPS.get(j % max(len(PERSONAS), 1), 0.9),
                             mode=args.mode, source_text=story)
        rec["prompt_id"] = pid
        rec["mode"] = args.mode
        rec["provider"] = prov["name"]
        return rec

    n_ok = n_short = n_fail = 0
    t0 = time.time()
    summary: dict = {}
    with shard.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_unit, u, k) for k, u in enumerate(todo)]
            for i, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if "error" in rec:
                    if rec["error"] == "too_short":
                        n_short += 1
                    else:
                        n_fail += 1
                        print(f"[wpgen] {rec['prompt_id']}:{rec['persona_idx']} "
                              f"failed: {rec['error']}", flush=True)
                else:
                    n_ok += 1
                if i % 25 == 0 or i == len(todo):
                    dt = max(time.time() - t0, 1e-9)
                    rate = i / dt
                    eta_s = (len(todo) - i) / max(rate, 1e-9)
                    eta = f"{int(eta_s // 3600):02d}:{int((eta_s % 3600) // 60):02d}"
                    print(f"[wpgen] ok={n_ok} too_short={n_short} fail={n_fail} "
                          f"rate/min={rate * 60:.1f} eta={eta}", flush=True)

    summary = {
        "providers": [p_["name"] for p_ in providers],
        "selected_pairs": len(units),
        "already_done": len(done),
        "generated_ok": n_ok,
        "too_short": n_short,
        "failed": n_fail,
        "elapsed_s": round(time.time() - t0, 1),
    }
    spath = OUT_DIR / "stories.summary.json"
    spath.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[wpgen] summary: ok={n_ok} too_short={n_short} failed={n_fail} "
          f"-> {spath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
