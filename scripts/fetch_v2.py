#!/usr/bin/env python3
"""Fetch + build the v2 training mix (pile-free, modern, length-balanced).

Sources (all HuggingFace unless noted):
  RAID             liamdugan/raid               — direct parquet shards (NO streaming; it stalls)
  M4 / SemEval-24  bitmind/sem-eval-24 (fallback yaful/M4)
  HC3              Hello-SimpleAI/HC3           — 5 configs, question-paired
  GPT-wiki-intro   aadityaubhat/GPT-wiki-intro  — topic-paired wiki intros
  Beemo            toloka/beemo                 — expert-edited human vs machine
  StoryScope       LOCAL data/raw/storyscope/*.parquet, else hf_hub_download
  v1 registers     coai, writingprompts, gutenberg, blogs from an existing train_all.parquet
                   (NOT pile/scp/storyscope)

Output (in --out-dir, default data/):
  v2_train.parquet                text,label,register,spans,+metadata cols
  v2_holdout.parquet              stratified 3% random
  v2_holdout_unseen_model.parquet all rows of one held-out generator
  v2_holdout_paraphrase.parquet   generation_method in {paraphrase,rewrite}, cap 5000
  v2_manifest.json                counts, licenses, revision pins, dedupe/balance stats

Build rules enforced: drop <200 chars, head-truncate 6000, dedupe
(exact-normalized + 120-char prefix), per-register length band
[q05,q95]*2 of the human distribution, any register capped at 25% of
total (small registers never killed entirely), binary labels only.

Usage:
  uv run python scripts/fetch_v2.py [--limit 500] [--skip-v1] [--out-dir data]

NETWORK PATHS ARE NOT RUNTIME-TESTED (offline machine); see
scripts/test_v2_smoke.py for what was exercised.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MIN_CHARS = 200
MAX_CHARS = 6000
PREFIX_LEN = 120

OUT_COLS = ["text", "label", "register", "spans",
            "source_dataset", "generator", "generation_method",
            "decoding", "split_hint"]

LICENSES = {
    "raid": "CC-BY?",
    "m4": "research",
    "hc3": "research",
    "wiki_intro": "MIT",
    "beemo": "CC-BY-SA",
    "storyscope": "MIT",
    "gutenberg": "public-domain",
    "blogs": "research-only",
    "coai": "research",
    "writingprompts": "public-scrape",
}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _dedup_key(t: str) -> str:
    return hashlib.md5(_norm(t)[:PREFIX_LEN].lower().encode()).hexdigest()


class Deduper:
    """Exact-full-text dedupe globally (true dups die regardless of label),
    plus 120-char-prefix near-dup dedupe WITHIN each label only.
    Cross-label near-twins (e.g. GPT-wiki-intro's generated intro starting
    like its human twin) are the training signal — they survive."""

    def __init__(self) -> None:
        self.seen_full: set[str] = set()
        self.seen_prefix: dict[int, set[str]] = defaultdict(set)

    def add(self, t: str, label: int | None = None) -> bool:
        full = hashlib.md5(t.encode()).hexdigest()
        if full in self.seen_full:
            return False
        self.seen_full.add(full)
        k = _dedup_key(t)
        lab = 0 if label is None else int(label)
        if k in self.seen_prefix[lab]:
            return False
        self.seen_prefix[lab].add(k)
        return True


def _mkrow(text: str, label: int, register: str, *, source: str, generator: str,
           method: str, decoding: str = "", hint: str = "") -> dict:
    return {
        "text": text,
        "label": int(label),
        "register": register,
        "spans": [],
        "source_dataset": source,
        "generator": generator,
        "generation_method": method,
        "decoding": decoding,
        "split_hint": hint,
    }


def clean_rows(rows: list[dict], dedupe: Deduper | None = None,
               stats: dict | None = None) -> list[dict]:
    """Drop short/empty, head-truncate, dedupe. Pure — runtime-testable."""
    out = []
    dropped = 0
    for r in rows:
        t = _norm(r["text"])
        if len(t) < MIN_CHARS:
            dropped += 1
            continue
        if dedupe is not None and not dedupe.add(t, int(r.get("label", 0))):
            dropped += 1
            continue
        r = dict(r)
        r["text"] = t[:MAX_CHARS]
        out.append(r)
    if stats is not None:
        stats["dropped_short_or_dup"] = stats.get("dropped_short_or_dup", 0) + dropped
    return out


def _hf_headers() -> dict:
    tok = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _http_json(url: str):
    import requests
    r = requests.get(url, headers=_hf_headers(), timeout=120)
    r.raise_for_status()
    return r.json()


def _dl_parquet(url: str):
    """Download one parquet from HF's auto-converted /parquet/ endpoint."""
    import pyarrow.parquet as pq
    import requests
    r = requests.get(url, headers=_hf_headers(), timeout=600)
    r.raise_for_status()
    return pq.read_table(io.BytesIO(r.content))


def _auto_urls(repo: str, config: str, split: str) -> list[str]:
    try:
        d = _http_json(f"https://huggingface.co/api/datasets/{repo}/parquet")
        return list(d.get(config, {}).get(split, []))
    except Exception as e:  # noqa: BLE001
        print(f"[v2] {repo}/{config}/{split}: no parquet branch ({e})", flush=True)
        return []


def fetch_raid(per_bucket: int = 150, shards: int = 4) -> tuple[list[dict], str]:
    """RAID train via the auto-converted parquet branch (raid/train/N.parquet),
    filter attack=='none', balanced sample per (domain, model), cap 150.
    Returns (rows, revision_sha_or_empty)."""
    from huggingface_hub import HfApi

    api = HfApi()
    sha = ""
    try:
        info = api.dataset_info("liamdugan/raid")
        sha = getattr(info, "sha", "") or ""
    except Exception as e:  # noqa: BLE001 — pin is best-effort
        print(f"[v2] raid revision pin failed: {e}", flush=True)
    urls = [f"https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/{i}.parquet"
            for i in range(shards)]
    print(f"[v2] raid: using {len(urls)} parquet-branch shards", flush=True)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for url in urls:
        try:
            table = _dl_parquet(url)
        except Exception as e:  # noqa: BLE001
            print(f"[v2] raid shard failed ({url.rsplit('/', 1)[-1]}): {e}", flush=True)
            continue
        df = table.to_pydict()
        attacks = df.get("attack") or []
        models = df.get("model") or []
        domains = df.get("domain") or []
        texts = df.get("generation") or df.get("text") or []
        decodes = df.get("decoding") or [""] * len(texts)
        for attack, model, domain, text, dec in zip(attacks, models, domains, texts, decodes):
            if attack not in ("none", None, ""):
                continue
            text = str(text or "")
            if not text:
                continue
            if model in ("none", "", None, "human"):
                key, gen = (str(domain), "human"), "human"
            else:
                key, gen = (str(domain), str(model)), str(model)
            b = buckets[key]
            if len(b) < per_bucket * 3:
                b.append(_mkrow(text, 0 if gen == "human" else 1,
                                f"raid_{domain}", source="raid", generator=gen,
                                method="human" if gen == "human" else "direct",
                                decoding=str(dec or ""), hint="raid:train-shard"))
    rng = random.Random(0)
    out = []
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        out.extend(items[:per_bucket])
    print(f"[v2] raid buckets={len(buckets)} rows={len(out)}", flush=True)
    return out, sha


def fetch_m4(per_domain: int = 8000) -> tuple[list[dict], str]:
    """SemEval-24 task8 subtask A from d0rj/SemEval2024-task8 (official data,
    parquet). label is a string '0'/'1'; domain col may be 'None' — use source
    then. Cap 8000/domain."""
    urls = _auto_urls("d0rj/SemEval2024-task8", "subtaskA_monolingual", "train")
    if not urls:
        print("[v2] m4 unavailable (no parquet branch)", flush=True)
        return [], ""
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for url in urls:
        df = _dl_parquet(url).to_pydict()
        texts = df.get("text") or []
        labs = df.get("label") or []
        models = df.get("model") or [""] * len(texts)
        domains = df.get("domain") or [""] * len(texts)
        sources = df.get("source") or [""] * len(texts)
        for text, lab, model, dom, src in zip(texts, labs, models, domains, sources):
            text = str(text or "")
            if not text:
                continue
            try:
                lab = int(str(lab))
            except (TypeError, ValueError):
                continue
            d = str(dom) if dom not in ("", "None", None) else (str(src) or "m4")
            if len(by_dom[d]) >= per_domain:
                continue
            gen = str(model) if (lab and model and str(model) != "None") else "unknown"
            by_dom[d].append(_mkrow(text, lab, f"m4_{d}".replace(" ", "_"),
                                    source="m4", generator="human" if lab == 0 else gen,
                                    method="human" if lab == 0 else "direct",
                                    hint=f"m4:{d}"))
    out = [r for dom in sorted(by_dom) for r in by_dom[dom]]
    print(f"[v2] m4 domains={len(by_dom)} rows={len(out)}", flush=True)
    return out, ""


def fetch_subtask_c(cap: int = 4000) -> list[dict]:
    """SemEval-24 subtask C: docs with mixed human+machine authorship
    (label==2). Eval-only slice — caller keeps it out of train."""
    urls = _auto_urls("d0rj/SemEval2024-task8", "subtaskC", "train")
    out = []
    for url in urls:
        df = _dl_parquet(url).to_pydict()
        for text, lab in zip(df.get("text") or [], df.get("label") or []):
            text = str(text or "")
            if not text:
                continue
            try:
                lab = int(str(lab))
            except (TypeError, ValueError):
                continue
            if lab != 2 or len(out) >= cap:
                continue
            out.append(_mkrow(text, 2, "semeval_mixed", source="semeval_c",
                              generator="mixed", method="mixed", hint="subtaskC"))
    print(f"[v2] subtaskC mixed rows={len(out)}", flush=True)
    return out


def fetch_hc3(limit_per_sub: int = 2500) -> tuple[list[dict], str]:
    """HC3 via auto-converted parquet branch. 2500/class/config. ChatGPT
    answers are 'rewrite' when the same question also has a human answer."""
    out = []
    for sub in ["reddit_eli5", "open_qa", "wiki_csai", "finance", "medicine"]:
        urls = _auto_urls("Hello-SimpleAI/HC3", sub, "train")
        if not urls:
            print(f"[v2] hc3/{sub} skipped: no parquet branch", flush=True)
            continue
        n_h = n_a = 0
        for url in urls:
            df = _dl_parquet(url).to_pydict()
            questions = df.get("question") or [""] * len(df.get("human_answers") or [])
            humans_col = df.get("human_answers") or []
            gpts_col = df.get("chatgpt_answers") or []
            for q, humans, gpts in zip(questions, humans_col, gpts_col):
                q = str(q or "")[:60]
                humans = humans or []
                gpts = gpts or []
                for a in humans:
                    if n_h >= limit_per_sub:
                        break
                    out.append(_mkrow(a, 0, f"hc3_{sub}", source="hc3", generator="human",
                                      method="human", hint=f"hc3:{q}"))
                    n_h += 1
                for a in gpts:
                    if n_a >= limit_per_sub:
                        break
                    method = "rewrite" if humans else "direct"
                    out.append(_mkrow(a, 1, f"hc3_{sub}_gpt", source="hc3",
                                      generator="chatgpt", method=method, hint=f"hc3:{q}"))
                    n_a += 1
        print(f"[v2] hc3/{sub}: h={n_h} ai={n_a}", flush=True)
    return out, ""


def fetch_wiki_intro(cap_each: int = 15000) -> tuple[list[dict], str]:
    """GPT-wiki-intro via parquet branch. Real columns: wiki_intro (human,
    label 0) + generated_intro (GPT-3.5, label 1); topic-paired."""
    urls = _auto_urls("aadityaubhat/GPT-wiki-intro", "default", "train")
    out = []
    n_h = n_a = 0
    for url in urls:
        df = _dl_parquet(url).to_pydict()
        wikis = df.get("wiki_intro") or []
        gens = df.get("generated_intro") or []
        titles = df.get("title") or [""] * len(wikis)
        for w, g, t in zip(wikis, gens, titles):
            hint = f"wiki_intro:{str(t or '')[:50]}"
            if n_h < cap_each and w:
                out.append(_mkrow(w, 0, "wiki_intro", source="wiki_intro",
                                  generator="human", method="human", hint=hint))
                n_h += 1
            if n_a < cap_each and g:
                out.append(_mkrow(g, 1, "wiki_intro_gpt", source="wiki_intro",
                                  generator="gpt-3.5", method="rewrite", hint=hint))
                n_a += 1
            if n_h >= cap_each and n_a >= cap_each:
                break
        if n_h >= cap_each and n_a >= cap_each:
            break
    print(f"[v2] wiki_intro: human={n_h} gpt={n_a}", flush=True)
    return out, ""


def fetch_beemo() -> tuple[list[dict], str]:
    """Beemo via parquet branch. Real columns: model_output (AI, one row per
    generator model), human_output (human). Edit-history cols ignored."""
    urls = _auto_urls("toloka/beemo", "default", "train")
    out = []
    for url in urls:
        df = _dl_parquet(url).to_pydict()
        models = df.get("model") or [""] * len(df.get("model_output") or [])
        for mo, ho, m in zip(df.get("model_output") or [], df.get("human_output") or [], models):
            if mo:
                out.append(_mkrow(mo, 1, "beemo_ai", source="beemo",
                                  generator=str(m or "unknown"), method="direct", hint="beemo"))
            if ho:
                out.append(_mkrow(ho, 0, "beemo", source="beemo",
                                  generator="human", method="human", hint="beemo"))
    print(f"[v2] beemo: {len(out)}", flush=True)
    return out, ""


STORY_COLS = [("story_gpt", "gpt"), ("story_deepseek", "deepseek"),
              ("story_kimi", "kimi"), ("story_gemini", "gemini"),
              ("story_claude", "claude")]


def _storyscope_file(split: str) -> Path | None:
    """Local copy if present, else hf_hub_download. Returns path or None."""
    from huggingface_hub import hf_hub_download

    local = ROOT / "data" / "raw" / "storyscope" / f"stories_{split}.parquet"
    if local.exists():
        return local
    try:
        return Path(hf_hub_download("jjrussell10/storyscope", f"stories_{split}.parquet",
                                    repo_type="dataset", local_dir=str(ROOT / "data" / "raw" / "storyscope")))
    except Exception as e:  # noqa: BLE001
        print(f"[v2] storyscope/{split} unavailable: {e}", flush=True)
        return None


def fetch_storyscope() -> list[dict]:
    """One AI row per non-null story_* column; generator set per column;
    prompt-level split kept in split_hint as 'storyscope_<split>#<prompt_id>'."""
    import pandas as pd

    out = []
    for split in ["train", "val", "test"]:
        p = _storyscope_file(split)
        if p is None:
            continue
        df = pd.read_parquet(p)
        pid_col = "prompt_id" if "prompt_id" in df.columns else None
        for rec in df.itertuples(index=False):
            rec_d = rec._asdict()
            pid = rec_d.get(pid_col, "") if pid_col else ""
            hint = f"storyscope_{split}#{pid}"
            for col, gen in STORY_COLS:
                text = rec_d.get(col)
                if text is None or (isinstance(text, float) and pd.isna(text)):
                    continue
                text = str(text).strip()
                if not text:
                    continue
                out.append(_mkrow(text, 1, "storyscope", source="storyscope",
                                  generator=gen, method="direct", hint=hint))
        print(f"[v2] storyscope/{split}: cumulative={len(out)}", flush=True)
    return out


V1_REGISTERS = ["coai", "writingprompts", "gutenberg", "blogs"]


def find_v1_parquet(explicit: Path | None) -> Path | None:
    """--v1-parquet wins; else data/train_all.parquet then ./train_all.parquet."""
    cands = [explicit] if explicit else [ROOT / "data" / "train_all.parquet", ROOT / "train_all.parquet"]
    for c in cands:
        if c and c.exists():
            return c
    return None


def load_v1_registers(parquet: Path, registers: list[str],
                      max_len: int = MAX_CHARS) -> list[dict]:
    """Pull selected registers from v1 train_all.parquet, head-truncated."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet)
    want = set(registers)
    out = []
    for batch in pf.iter_batches(batch_size=100_000,
                                 columns=["text", "label", "register"]):
        df = batch.to_pandas()
        df = df[df.register.isin(want)]
        for t, l, r in zip(df["text"], df["label"], df["register"]):
            out.append(_mkrow(str(t)[:max_len], int(l), str(r), source=str(r),
                              generator="human" if int(l) == 0 else "unknown",
                              method="human" if int(l) == 0 else "direct",
                              hint=f"v1:{r}"))
    return out


def balance_lengths(rows: list[dict]) -> tuple[list[dict], dict]:
    """Per register: drop rows outside [q05,q95]*2 of the HUMAN length
    distribution (registers without a human side pass through untouched)."""
    by_reg: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_reg[r["register"]][int(r["label"])].append(len(r["text"]))
    bands: dict[str, tuple[int, int]] = {}
    for reg, labs in by_reg.items():
        if 0 not in labs:
            continue
        h = sorted(labs[0])
        q05, q95 = h[int(len(h) * 0.05)], h[min(int(len(h) * 0.95), len(h) - 1)]
        bands[reg] = (min(q05, MIN_CHARS), max(q95, 1200))
    out, dropped = [], Counter()
    for r in rows:
        band = bands.get(r["register"])
        if band is None:
            out.append(r)
            continue
        lo, hi = band
        if lo <= len(r["text"]) <= hi * 2:
            out.append(r)
        else:
            dropped[r["register"]] += 1
    return out, {k: int(v) for k, v in dropped.items()}


def cap_register(rows: list[dict], frac: float = 0.25) -> tuple[list[dict], dict]:
    """Cap any register at frac of total; small registers never killed
    entirely (floor of 1000 rows per register)."""
    counts = Counter(r["register"] for r in rows)
    total = sum(counts.values())
    cap_for = {reg: max(int(total * frac), 1000) for reg in counts}
    order = sorted(range(len(rows)), key=lambda i: random.Random(1).random())
    used: Counter[str] = Counter()
    out = []
    for i in order:
        reg = rows[i]["register"]
        if used[reg] < cap_for[reg]:
            out.append(rows[i])
            used[reg] += 1
    trimmed = {reg: counts[reg] - used[reg] for reg in counts if used[reg] < counts[reg]}
    return out, trimmed


def stratified_holdout(rows: list[dict], frac: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """~frac of FAMILY GROUPS per (register, label) stratum. A family is a
    paired group (shared hint prefix before '#', e.g. hc3:<question> or
    wiki_intro:<title> or storyscope_<split>#<prompt_id>) so twins never
    straddle train/holdout. Unpaired rows are their own family."""
    rng = random.Random(seed)

    # paired sources whose hint is a per-pair key; everything else is row-level
    PAIRED_SOURCES = {"hc3", "wiki_intro"}

    def family(i: int, r: dict) -> str:
        h = r.get("split_hint") or ""
        if "#" in h:  # storyscope_train#<prompt_id> — per-prompt family
            return h
        if r.get("source_dataset") in PAIRED_SOURCES:  # hc3:<q> / wiki_intro:<title>
            return f"{r['source_dataset']}:{h}"
        return f"__row__{i}"  # unpaired source — each row its own family

    fams: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        fams[(r["register"], int(r["label"]), family(i, r))].append(i)
    # group families per (register,label) then sample families
    strata: dict[tuple[str, int], list[tuple[str, list[int]]]] = defaultdict(list)
    for (reg, lab, fam), idxs in fams.items():
        strata[(reg, lab)].append((fam, idxs))
    hold_idx: set[int] = set()
    for idxs_groups in strata.values():
        total = sum(len(g) for _, g in idxs_groups)
        k = max(1, int(round(total * frac)))
        rng.shuffle(idxs_groups)
        got = 0
        for _, g in idxs_groups:
            if got >= k:
                break
            hold_idx.update(g)
            got += len(g)
    hold = [rows[i] for i in sorted(hold_idx)]
    rest = [r for i, r in enumerate(rows) if i not in hold_idx]
    return hold, rest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-parquet", type=Path, default=None,
                    help="path to v1 train_all.parquet (default: try data/train_all.parquet then ./)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--limit", type=int, default=0, help="global smoke cap: max rows per source")
    ap.add_argument("--skip-v1", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    deduper = Deduper()
    manifest: dict = {"sources": {}, "licenses": LICENSES, "revisions": {},
                      "dedupe": {}, "length_balance": {}, "caps": {}}
    dedupe_stats: dict = {}
    lim = args.limit

    def take(rows: list[dict], name: str) -> list[dict]:
        if lim:
            rows = rows[:lim]
        manifest["sources"][name] = len(rows)
        print(f"[v2] {name}: {len(rows)} rows fetched", flush=True)
        return rows

    print("[v2] fetching RAID (direct parquet shards, no streaming)...", flush=True)
    rows, sha = fetch_raid()
    manifest["revisions"]["raid"] = sha
    rows = clean_rows(take(rows, "raid"), deduper, dedupe_stats)

    print("[v2] fetching M4/SemEval-24...", flush=True)
    rows += clean_rows(take(fetch_m4()[0], "m4"), deduper, dedupe_stats)

    print("[v2] fetching HC3...", flush=True)
    rows += clean_rows(take(fetch_hc3()[0], "hc3"), deduper, dedupe_stats)

    print("[v2] fetching GPT-wiki-intro...", flush=True)
    rows += clean_rows(take(fetch_wiki_intro()[0], "wiki_intro"), deduper, dedupe_stats)

    print("[v2] fetching Beemo...", flush=True)
    rows += clean_rows(take(fetch_beemo()[0], "beemo"), deduper, dedupe_stats)

    # StoryScope dropped in v2: AI-only, anthology prompts, no human pairs.
    # Methodology copied instead: fetch_wp_prompts.py + generate_ai_stories.py
    # produce fictpair rows (same Reddit prompt, human story vs ox-alpha stories).
    print("[v2] storyscope SKIPPED (replaced by fictpair generation)", flush=True)

    # mixed-authorship docs: eval-only, never in train
    mixed = clean_rows(fetch_subtask_c(), deduper, dedupe_stats) if not lim else []
    if mixed:
        import pandas as pd
        mp = args.out_dir / "v2_holdout_mixed.parquet"
        pd.DataFrame(mixed).to_parquet(mp, index=False)
        manifest["sources"]["semeval_mixed_holdout"] = len(mixed)
        print(f"[v2] wrote {mp} ({len(mixed)} mixed-authorship eval rows)", flush=True)

    if not args.skip_v1:
        v1p = find_v1_parquet(args.v1_parquet)
        if v1p:
            print(f"[v2] pulling v1 registers {V1_REGISTERS} from {v1p}...", flush=True)
            rows += clean_rows(take(load_v1_registers(v1p, V1_REGISTERS), "v1_registers"),
                               deduper, dedupe_stats)
        else:
            print("[v2] no v1 parquet found, skipping", flush=True)

    manifest["dedupe"] = dedupe_stats
    print(f"[v2] after clean+dedupe: {len(rows)}", flush=True)

    print("[v2] balancing lengths per register ([q05,q95]*2 of human)...", flush=True)
    rows, bal = balance_lengths(rows)
    manifest["length_balance"] = bal
    print(f"[v2] after length balance: {len(rows)} (dropped {sum(bal.values())})", flush=True)

    print("[v2] capping registers at 25%...", flush=True)
    rows, trimmed = cap_register(rows)
    manifest["caps"] = trimmed
    print(f"[v2] after cap: {len(rows)} (trimmed {sum(trimmed.values())})", flush=True)

    assert all(r["label"] in (0, 1) for r in rows), "binary labels only"

    # holdout (b): ALL rows of the rarest RAID generator present
    raid_models = Counter(r["generator"] for r in rows
                          if r["source_dataset"] == "raid" and r["generator"] != "human")
    unseen = min(raid_models, key=lambda m: raid_models[m]) if raid_models else ""
    unseen_rows = [r for r in rows if unseen and r["generator"] == unseen]
    rows = [r for r in rows if not (unseen and r["generator"] == unseen)]
    manifest["unseen_model"] = {"generator": unseen, "rows": len(unseen_rows)}
    print(f"[v2] unseen-model holdout: '{unseen}' ({len(unseen_rows)} rows)", flush=True)

    # holdout (c): small slice of paraphrase/rewrite rows (10%, cap 1000).
    # the rest STAY IN TRAIN — rewrites are training signal, not contamination.
    para_pool = [r for r in rows if r["generation_method"] in ("paraphrase", "rewrite")]
    random.Random(7).shuffle(para_pool)
    para_rows = para_pool[:min(max(int(len(para_pool) * 0.1), 1), 1000)] if para_pool else []
    para_set = {id(r) for r in para_rows}
    rows = [r for r in rows if id(r) not in para_set]
    print(f"[v2] paraphrase/rewrite holdout: {len(para_rows)} of {len(para_pool)} "
          f"(rest stay in train)", flush=True)

    # holdout (a): stratified 3%
    hold_rows, train = stratified_holdout(rows, 0.03)
    print(f"[v2] stratified holdout: {len(hold_rows)}; train: {len(train)}", flush=True)

    counts = Counter((r["register"], r["label"]) for r in train)
    manifest["final_train_counts"] = {f"{reg}|{lab}": n for (reg, lab), n in sorted(counts.items())}
    print("[v2] final mix:")
    for (reg, lab), n in sorted(counts.items()):
        print(f"   {reg:24s} label={lab} {n:7d}")

    import pandas as pd

    out = args.out_dir
    pd.DataFrame(train)[OUT_COLS].to_parquet(out / "v2_train.parquet", index=False)
    pd.DataFrame(hold_rows)[OUT_COLS].to_parquet(out / "v2_holdout.parquet", index=False)
    pd.DataFrame(unseen_rows)[OUT_COLS].to_parquet(out / "v2_holdout_unseen_model.parquet", index=False)
    pd.DataFrame(para_rows)[OUT_COLS].to_parquet(out / "v2_holdout_paraphrase.parquet", index=False)
    (out / "v2_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[v2] wrote v2_train ({len(train)}), 3 holdouts, manifest -> {out}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
