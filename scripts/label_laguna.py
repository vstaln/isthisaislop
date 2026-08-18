#!/usr/bin/env python3
"""Label texts with local ontology hits + Laguna style/construction JSON.

Reads TOKENROUTER_* (preferred), then OPENROUTER_*, then COMMAND_CODE_* from .env.
incrementally so a 503 does not wipe the batch. Never writes the API key
into the output.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slopdet.construction import construction_stats
from slopdet.explain import REGISTER_IDS, explain
from slopdet.ontology import load_ontology
from slopdet.tags import hydrate_lanes, pack_style
from slopdet.weaklabel import label_text

OUT = ROOT / "eval" / "labels" / "laguna.jsonl"
SYS = """Tag checkable AI-slop features. Never claim authorship. Never percent-AI.

Emit ONLY an id from this list, plus a real quote. No reasons. No other keys.

style (slop): glue frames emdash cliche puffery opener weasel colon
  glue = delve/leverage/foster/utilize/empower/streamline
  puffery = robust/seamless/pivotal/tapestry/cutting-edge
  opener = here's the thing / as an AI
  frames = in today's / it's worth noting / at its core / in conclusion
  weasel = experts agree / studies show
  colon = The best part: it learns
  emdash = em-dash cluster
  cliche = heart hammered / voice barely a whisper / took a deep breath

construction slop: moral gloss recap even
  moral = narrator states the lesson
  gloss = in other words / the point is / as you can see
  recap = tidy wrap-up after the last fact
  even = same sentence skeleton repeated

construction human: jump anchor open burst
  jump = flashback / years earlier / meanwhile
  anchor = weekday, clock, named place/person, number
  open = maybe / not sure / still unresolved
  burst = short sentence next to a much longer one

Return ONLY JSON:
{"lean":"slop|human|mixed|unclear","style":[{"id":"glue","quote":"..."}],"construction":[{"id":"anchor","quote":"..."}]}
Empty lanes are [].
"""


def load_env() -> dict[str, str]:
    raw: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        raise SystemExit(
            "missing .env with TOKENROUTER_API_KEY, OPENROUTER_API_KEY, or COMMAND_CODE_API_KEY"
        )
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        raw[key.strip()] = val.strip()
    if raw.get("COMMAND_CODE_API_KEY"):
        return {
            "API_KEY": raw["COMMAND_CODE_API_KEY"],
            "MODEL": raw.get("COMMAND_CODE_MODEL", "poolside/laguna-s-2.1-free"),
            "BASE": raw.get("COMMAND_CODE_BASE", "https://api.commandcode.ai/provider/v1"),
            "HTTP_REFERER": "",
            "X_TITLE": "",
        }
    if raw.get("TOKENROUTER_API_KEY"):
        return {
            "API_KEY": raw["TOKENROUTER_API_KEY"],
            "MODEL": raw.get("TOKENROUTER_MODEL", "deepseek/deepseek-v4-pro-0813-free"),
            "BASE": raw.get("TOKENROUTER_BASE", "https://api.tokenrouter.com/v1"),
            "HTTP_REFERER": "",
            "X_TITLE": "",
        }
    if raw.get("OPENROUTER_API_KEY"):
        return {
            "API_KEY": raw["OPENROUTER_API_KEY"],
            "MODEL": raw.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
            "BASE": raw.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1"),
            "HTTP_REFERER": raw.get("OPENROUTER_HTTP_REFERER", "https://github.com/isthisaislop"),
            "X_TITLE": raw.get("OPENROUTER_X_TITLE", "ITAIS"),
        }
    if raw.get("COMMAND_CODE_API_KEY"):
        return {
            "API_KEY": raw["COMMAND_CODE_API_KEY"],
            "MODEL": raw.get("COMMAND_CODE_MODEL", "poolside/laguna-s-2.1-free"),
            "BASE": raw.get("COMMAND_CODE_BASE", "https://api.commandcode.ai/provider/v1"),
            "HTTP_REFERER": "",
            "X_TITLE": "",
        }
    raise SystemExit(
        "TOKENROUTER_API_KEY, OPENROUTER_API_KEY, or COMMAND_CODE_API_KEY missing"
    )


def doc_id(text: str, prefix: str) -> str:
    return prefix + hashlib.sha256(text.encode()).hexdigest()[:16]


def already_done(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.is_file():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("laguna"):
            seen.add(rec["id"])
    return seen


def local_hits(text: str, onto) -> list[dict]:
    hits = []
    for hit in label_text(text, onto):
        if hit["id"] in REGISTER_IDS:
            continue
        hits.append(pack_style(hit))
    return hits


def laguna(env: dict[str, str], text: str, retries: int = 12) -> dict:
    payload = {
        "model": env["MODEL"],
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": text[:6000]},
        ],
    }
    if "deepseek" in env["MODEL"].lower():
        payload["thinking"] = {"type": "disabled"}
    req_path = Path("/tmp/itais_laguna_req.json")
    out_path = Path("/tmp/itais_laguna_resp.json")
    hdr_path = Path("/tmp/itais_laguna_hdr")
    req_path.write_text(json.dumps(payload), encoding="utf-8")
    hdr = f"Authorization: Bearer {env['API_KEY']}\nContent-Type: application/json\n"
    if env.get("HTTP_REFERER"):
        hdr += f"HTTP-Referer: {env['HTTP_REFERER']}\n"
    if env.get("X_TITLE"):
        hdr += f"X-Title: {env['X_TITLE']}\n"
    hdr_path.write_text(hdr, encoding="utf-8")
    curl = [
        "curl",
        "-sS",
        "-o",
        str(out_path),
        "-w",
        "%{http_code}",
        "-H",
        f"@{hdr_path}",
        f"{env['BASE']}/chat/completions",
        "-d",
        f"@{req_path}",
    ]
    last = None
    nudges = 0
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                curl,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            print(f"    laguna timeout attempt {attempt + 1}/{retries}", flush=True)
            time.sleep(min(45, 10 + attempt * 10))
            continue
        code = proc.stdout.strip()
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"error": {"message": out_path.read_text(encoding="utf-8")[:400]}}
        last = (code, data)
        if code == "200" and data.get("choices"):
            msg = data["choices"][0].get("message") or {}
            raw = msg.get("content")
            if not raw:
                raw = msg.get("reasoning_content")
            if isinstance(raw, list):
                raw = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw
                )
            raw = "" if raw is None else str(raw)
            if "</think>" in raw:
                raw = raw.split("</think>", 1)[-1]
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                if nudges < 2:
                    nudges += 1
                    print(f"    laguna empty_json nudge {nudges}", flush=True)
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                "Your reply was not JSON. Reply with ONLY the JSON "
                                "object, no prose, no fences."
                            ),
                        }
                    )
                    req_path.write_text(json.dumps(payload), encoding="utf-8")
                    time.sleep(3)
                    continue
                print(f"    laguna {code} empty_json attempt {attempt + 1}/{retries}", flush=True)
                time.sleep(min(45, 10 + attempt * 10))
                continue
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                if nudges < 2:
                    nudges += 1
                    print(f"    laguna bad_json nudge {nudges}", flush=True)
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                "Your JSON did not parse. Reply with ONLY valid JSON "
                                "matching the schema, no prose, no fences."
                            ),
                        }
                    )
                    req_path.write_text(json.dumps(payload), encoding="utf-8")
                    time.sleep(3)
                    continue
                print(f"    laguna {code} bad_json attempt {attempt + 1}/{retries}", flush=True)
                time.sleep(min(45, 10 + attempt * 10))
                continue
            parsed = hydrate_lanes(parsed)
            if not parsed["style"] and not parsed["construction"] and nudges < 2:
                nudges += 1
                print(f"    laguna empty_lanes nudge {nudges}", flush=True)
                payload["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            "Your reply had no tagged spans. Reply with ONLY the JSON. "
                            "Put each real id from the list in style or construction "
                            "with a verbatim quote from the text."
                        ),
                    }
                )
                req_path.write_text(json.dumps(payload), encoding="utf-8")
                time.sleep(3)
                continue
            parsed["_usage"] = data.get("usage")
            return parsed
        err = (data.get("error") or {}).get("type") or (data.get("error") or {}).get("message") or code
        print(f"    laguna {code} {err} attempt {attempt + 1}/{retries}", flush=True)
        time.sleep(min(45, 10 + attempt * 10))
    raise RuntimeError(f"laguna failed: {last[0]} {last[1]}")


def sample_coai(n_per_class: int, pile: int | None = None, seed: int = 1) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(ROOT / "data" / "coai_train.parquet")
    rows: list[dict] = []
    if pile is None or pile == 0:
        humans = df[df["label"] == 0].sample(n_per_class, random_state=seed)
        for rec in humans.itertuples(index=False):
            rows.append(
                {
                    "id": doc_id(str(rec.text), "coai-h-"),
                    "pile": 0,
                    "model": str(rec.model_name),
                    "source": "coai_train",
                    "text": str(rec.text),
                }
            )
    if pile is None or pile == 1:
        ai = df[df["label"] == 1]
        models = [m for m in ai["model_name"].unique() if m != "Human"]
        per = max(1, n_per_class // max(len(models), 1))
        leftover = n_per_class
        for model in models:
            chunk = ai[ai["model_name"] == model]
            take = min(per, leftover, len(chunk))
            chunk = chunk.sample(take, random_state=seed)
            leftover -= take
            for rec in chunk.itertuples(index=False):
                rows.append(
                    {
                        "id": doc_id(str(rec.text), "coai-a-"),
                        "pile": 1,
                        "model": str(rec.model_name),
                        "source": "coai_train",
                        "text": str(rec.text),
                    }
                )
    return rows


def seed_docs() -> list[dict]:
    path = ROOT / "data" / "corpus.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        rows.append(
            {
                "id": rec.get("id") or doc_id(rec["text"], "seed-"),
                "pile": int(rec.get("pile", 0)),
                "model": rec.get("model", "seed"),
                "source": "seed",
                "text": rec["text"],
            }
        )
    return rows


def verbatim(doc_text: str, hits: list[dict]) -> list[dict]:
    """Keep only hits whose quote is literally in the doc. Fabricated quotes die."""
    return [h for h in hits if h.get("quote") and h["quote"] in doc_text]


def main() -> None:
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    pile = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    env = load_env()
    onto = load_ontology()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = already_done(OUT)
    docs = seed_docs() + sample_coai(n_per, pile, seed)
    todo = [d for d in docs if d["id"] not in seen]
    print(f"todo {len(todo)} already {len(seen)} out {OUT} model={env['MODEL']} base={env['BASE']}", flush=True)
    ok = fail = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for i, doc in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] pile={doc['pile']} {doc['source']} {doc['model']}", flush=True)
            try:
                named = local_hits(doc["text"], onto)
                llm = laguna(env, doc["text"])
                llm["style"] = verbatim(doc["text"], llm.get("style") or [])
                llm["construction"] = verbatim(doc["text"], llm.get("construction") or [])
                rec = {
                    "id": doc["id"],
                    "pile": doc["pile"],
                    "model": doc["model"],
                    "source": doc["source"],
                    "text": doc["text"],
                    "local_style_hits": named,
                    "construction_stats": construction_stats(doc["text"]),
                    "local": explain(doc["text"]),
                    "laguna": llm,
                    "labeler": env["MODEL"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                ok += 1
                print(
                    f"    lean={llm.get('lean')} style={len(llm.get('style') or [])} "
                    f"construction={len(llm.get('construction') or [])} local={len(named)}",
                    flush=True,
                )
                time.sleep(2)
            except Exception as exc:  # noqa: BLE001 — keep going, log the miss
                fail += 1
                rec = {
                    "id": doc["id"],
                    "pile": doc["pile"],
                    "model": doc["model"],
                    "source": doc["source"],
                    "text": doc["text"],
                    "local_style_hits": local_hits(doc["text"], onto),
                    "construction_stats": construction_stats(doc["text"]),
                    "laguna": None,
                    "laguna_error": type(exc).__name__,
                    "labeler": env["MODEL"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"    FAIL {type(exc).__name__}", flush=True)
    print(f"done ok={ok} fail={fail} file={OUT}", flush=True)


if __name__ == "__main__":
    main()
