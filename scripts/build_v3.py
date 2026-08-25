"""Build v3 from v2: kill the measured label shortcuts, keep the provenance.

v2, MEASURED (artifacts/corpus_probes.json): register-only AUROC 0.989, length-only 0.705,
81% of docs in single-label registers. A model trained on it learns dataset-of-origin, not writing.

Fixes, in order:
  F1  m4 refetched with a per-(domain,label) cap — v2's per-domain cap filled from a label-ordered
      file and produced five 100%-AI registers.
  F2  register = the text's register, not its provenance. `*_gpt`/`beemo_ai` twins merge into their
      human register; `rewrite_pair`/`respond_pair` rows inherit their human seed's register via the
      split_hint pair-key (exact join first, seed-name fallback).
  F3  length matching: per register, the longer class is sentence-truncated to targets sampled from
      the shorter class's length distribution (global AI distribution for single-label registers).
  F4  era tag per row; post-2022 verified-human is still absent and stays a named gap for D1.

Also writes a *balanced* view (per-register majority downsample) — that is the training file.

    uv run python scripts/build_v3.py --v2 /tmp/v2_train.parquet --out data/v3
    uv run python scripts/build_v3.py --holdouts-only   # rename registers in the 4 holdout files
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from corpus_probes import auroc, two_sided  # noqa: E402

HF = "https://huggingface.co/datasets/vstalingrady/itais/resolve/main/{name}"
HOLDOUTS = ["v2_holdout_labeled.parquet", "v2_holdout_paraphrase_labeled.parquet",
            "v2_holdout_mixed.parquet", "v2_holdout_unseen_model_labeled.parquet"]
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n{2,}")
SEED_FALLBACK = {"hc3x": "hc3_mixed"}
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- F2: registers

def canonicalize_registers(df: pd.DataFrame) -> pd.DataFrame:
    reg = df.register.copy()
    reg = reg.str.replace(r"_gpt$", "", regex=True)
    reg = reg.replace({"beemo_ai": "beemo"})

    human_hints = df[(df.label == 0) & df.split_hint.str.startswith("para:", na=False)]
    hint2reg = dict(zip(human_hints.split_hint, human_hints.register.str.replace(r"_gpt$", "", regex=True)))

    def rewrite_reg(hint: str) -> str:
        if hint in hint2reg:
            return hint2reg[hint]
        seed = hint.split(":")[1] if ":" in hint else "unknown"
        return SEED_FALLBACK.get(seed, seed)

    mask = df.register == "rewrite_pair"
    reg.loc[mask] = df.loc[mask, "split_hint"].map(rewrite_reg)
    reg.loc[df.register == "respond_pair"] = "writingprompts"
    out = df.copy()
    out["register"] = reg
    return out


# ---------------------------------------------------------------- F1: m4 refetch

def refetch_m4(per_cell: int = 4000) -> pd.DataFrame:
    from fetch_v2 import _auto_urls, _dl_parquet, _mkrow  # noqa: PLC0415

    urls = _auto_urls("d0rj/SemEval2024-task8", "subtaskA_monolingual", "train")
    if not urls:
        raise SystemExit("m4 parquet branch unavailable")
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for url in urls:
        cols = _dl_parquet(url).to_pydict()
        n = len(cols.get("text") or [])
        for i in range(n):
            text = str(cols["text"][i] or "")
            if not text:
                continue
            try:
                lab = int(str(cols["label"][i]))
            except (TypeError, ValueError):
                continue
            dom = str(cols.get("domain", [""] * n)[i] or "") or str(cols.get("source", [""] * n)[i] or "m4")
            if dom in ("", "None"):
                dom = "m4"
            if len(cells[(dom, lab)]) >= per_cell:
                continue
            model = str(cols.get("model", [""] * n)[i] or "")
            cells[(dom, lab)].append(_mkrow(
                text, lab, f"m4_{dom}".replace(" ", "_"), source="m4",
                generator="human" if lab == 0 else (model if model not in ("", "None") else "unknown"),
                method="human" if lab == 0 else "direct", hint=f"m4:{dom}"))
    rows = [r for key in sorted(cells) for r in cells[key]]
    df = pd.DataFrame(rows)
    df = df[~df.text.duplicated()]
    print(f"[m4] refetched {len(df)} rows, "
          f"human {int((df.label == 0).sum())} / ai {int((df.label == 1).sum())}", flush=True)
    return df


# ---------------------------------------------------------------- F3: lengths

def truncate_at_sentence(text: str, target_words: int) -> str:
    if len(text.split()) <= target_words:
        return text
    sentences = SENT_SPLIT.split(text)
    out, count = [], 0
    for sentence in sentences:
        w = len(sentence.split())
        if out and count + w > target_words:
            break
        out.append(sentence)
        count += w
    kept = " ".join(out).strip()
    return kept if count >= max(30, int(target_words * 0.5)) else " ".join(text.split()[:target_words])


def match_lengths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_words"] = df.text.str.split().str.len()
    global_ai = df[df.label == 1].n_words.values
    new_text = df.text.copy()
    for register, group in df.groupby("register"):
        by_label = {lab: sub for lab, sub in group.groupby("label")}
        if len(by_label) == 2:
            med0 = by_label[0].n_words.median()
            med1 = by_label[1].n_words.median()
            short_label = 0 if med0 <= med1 else 1
            targets_pool = by_label[short_label].n_words.values
            victims = by_label[1 - short_label]
        else:
            targets_pool = global_ai
            victims = group
        targets = RNG.choice(targets_pool, size=len(victims))
        for (idx, row), target in zip(victims.iterrows(), targets):
            target = int(min(max(target, 40), 700))
            if row.n_words > target * 1.1:
                new_text.at[idx] = truncate_at_sentence(row.text, target)
    df["text"] = new_text
    df["n_words"] = df.text.str.split().str.len()
    df = df[df.n_words >= 30].drop(columns=["n_words"])
    df["spans"] = [[] for _ in range(len(df))]  # offsets invalidated by truncation; span labels are D2's job
    return df


# ---------------------------------------------------------------- balanced view

def balance(df: pd.DataFrame, ratio: int = 1, floor: int = 1500) -> pd.DataFrame:
    kept = []
    for register, group in df.groupby("register"):
        n0, n1 = int((group.label == 0).sum()), int((group.label == 1).sum())
        if not n0 or not n1:
            kept.append(group.sample(min(len(group), floor), random_state=0))
            continue
        minority = 0 if n0 <= n1 else 1
        cap = max(ratio * min(n0, n1), floor)
        kept.append(group[group.label == minority])
        kept.append(group[group.label != minority].sample(
            min(max(n0, n1), cap), random_state=0))
    return pd.concat(kept).sample(frac=1.0, random_state=0).reset_index(drop=True)


# ---------------------------------------------------------------- probes

def probe(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["n_words"] = df.text.str.split().str.len()
    per = df.groupby("register").label.agg(n="size", ai="mean")
    df["p_reg"] = df.register.map(per.ai)
    single = per[(per.ai <= 0.02) | (per.ai >= 0.98)]
    return {
        "rows": len(df),
        "registers": len(per),
        "single_label_registers": len(single),
        "share_single_label": round(float(single.n.sum() / len(df)), 3),
        "register_only_auroc": round(two_sided(auroc(df.p_reg.values, df.label.values)), 3),
        "length_only_auroc": round(two_sided(auroc(df.n_words.values, df.label.values)), 3),
        "median_words": {"human": float(df[df.label == 0].n_words.median()),
                         "ai": float(df[df.label == 1].n_words.median())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, default=Path("/tmp/v2_train.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/v3"))
    ap.add_argument("--holdouts-only", action="store_true")
    ap.add_argument("--skip-m4-refetch", action="store_true")
    ap.add_argument("--refine", action="store_true", help="reload v3_train_full, re-match lengths, rebalance, re-probe")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"built_from": str(args.v2)}

    if args.refine:
        df = pd.read_parquet(args.out / "v3_train_full.parquet")
        df = match_lengths(df)
        report["v3_full_probes"] = probe(df)
        bal = balance(df)
        report["v3_balanced_probes"] = probe(bal)
        report["per_register_balanced"] = {
            k: {"n": int(v["size"]), "ai": round(float(v["mean"]), 3)}
            for k, v in bal.groupby("register").label.agg(["size", "mean"]).iterrows()}
        df.to_parquet(args.out / "v3_train_full.parquet", index=False)
        bal.to_parquet(args.out / "v3_train_balanced.parquet", index=False)
        Path("artifacts/corpus_probes_v3.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0

    if not args.holdouts_only:
        if not args.v2.exists():
            urllib.request.urlretrieve(HF.format(name="v2_train_labeled.parquet"), args.v2)
        df = pd.read_parquet(args.v2)
        report["v2_probes"] = probe(df)

        df = canonicalize_registers(df)
        if not args.skip_m4_refetch:
            df = pd.concat([df[~df.register.str.startswith("m4_")], refetch_m4()], ignore_index=True)
        df["era"] = np.where(df.label == 0, "pre", "post")
        df = match_lengths(df)
        report["v3_full_probes"] = probe(df)

        bal = balance(df)
        report["v3_balanced_probes"] = probe(bal)
        report["per_register_balanced"] = {
            k: {"n": int(v["size"]), "ai": round(float(v["mean"]), 3)}
            for k, v in bal.groupby("register").label.agg(["size", "mean"]).iterrows()}

        df.to_parquet(args.out / "v3_train_full.parquet", index=False)
        bal.to_parquet(args.out / "v3_train_balanced.parquet", index=False)

    for name in HOLDOUTS:
        path = Path("/tmp") / name
        try:
            if not path.exists():
                urllib.request.urlretrieve(HF.format(name=name), path)
            h = canonicalize_registers(pd.read_parquet(path))
            h["era"] = np.where(h.label == 0, "pre", "post")
            h.to_parquet(args.out / name.replace("v2_", "v3_"), index=False)
            report.setdefault("holdouts", []).append(name.replace("v2_", "v3_"))
        except Exception as exc:  # noqa: BLE001
            report.setdefault("holdout_errors", {})[name] = f"{type(exc).__name__}: {exc}"

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/corpus_probes_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_register_balanced"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
