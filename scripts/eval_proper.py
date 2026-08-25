#!/usr/bin/env python3
"""Proper v1 eval: cross-register AUROC matrix + held-out generalization files.

Runs on the VM after full training. Reuses eval_trained.DetectorBundle + fine_tune_lfm
tokenizer path so scoring is identical to the shipped eval.

Outputs artifacts/eval_proper/report.json with:
  - per-register score stats (mean/median per class present)
  - cross-register AUROC matrix: every human register x every AI register
  - held-out jsonl files (eval/labels/*.jsonl) AUROC each
  - coai strict: val coai rows AUROC (vs regex floor 0.8643)

Usage:
  python3 scripts/eval_proper.py --ckpt artifacts/lfm/model.pt \
      --spans-parquet train_all.parquet --out artifacts/eval_proper
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from transformers import AutoTokenizer  # noqa: E402

from eval_trained import DetectorBundle, auroc  # noqa: E402
from slopdet.lfm import load_encoder_body  # noqa: E402

HUMAN_REGS = ["writingprompts", "gutenberg", "blogs"]
AI_REGS = ["storyscope", "coai", "pile"]
HELD_OUT = ["eval/labels/laguna.jsonl", "eval/labels/local.jsonl", "eval/labels/deepseek_eval.jsonl"]


def load_rows_minimal(parquet: str, registers: list[str] | None = None, cap_per_reg: int = 3000):
    """Stream parquet, return [(text, label, register)] capped per register."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet)
    want = set(registers) if registers else None
    counts: dict[str, int] = defaultdict(int)
    out = []
    for batch in pf.iter_batches(batch_size=100000, columns=["text", "label", "register"]):
        df = batch.to_pandas()
        for t, l, r in zip(df["text"], df["label"], df["register"]):
            if want is not None and r not in want:
                continue
            if counts[r] >= cap_per_reg:
                continue
            counts[r] += 1
            out.append((str(t), int(l), str(r)))
        if want and all(counts[r] >= cap_per_reg for r in want):
            break
    return out


@torch.no_grad()
def score_rows(model, tok, rows, device, max_len: int, batch_size: int = 32):
    scores, labels, regs = [], [], []
    model.eval()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = tok([r[0] for r in chunk], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        doc_logits, _ = model(enc["input_ids"], enc["attention_mask"])
        prob = torch.softmax(doc_logits, -1)[:, 1].float().cpu().tolist()
        scores.extend(prob)
        labels.extend(r[1] for r in chunk)
        regs.extend(r[2] for r in chunk)
    return scores, labels, regs


def stats(scores, labels, regs):
    by: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s, y, r in zip(scores, labels, regs):
        by[r][y].append(s)
    out = {}
    for r, labs in by.items():
        entry = {"n": sum(len(v) for v in labs.values())}
        for y, v in labs.items():
            v_sorted = sorted(v)
            entry[f"mean_{'ai' if y else 'human'}"] = round(sum(v) / len(v), 4)
            entry[f"med_{'ai' if y else 'human'}"] = v_sorted[len(v)//2]
        out[r] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/lfm/model.pt")
    ap.add_argument("--spans-parquet", default="train_all.parquet")
    ap.add_argument("--model", default="LiquidAI/LFM2.5-Encoder-350M")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--cap", type=int, default=3000, help="max rows per register")
    ap.add_argument("--out", default="artifacts/eval_proper")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[proper] device={device} ckpt={args.ckpt}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = DetectorBundle(args.model, n_lanes=4).to(device)  # v1 lanes: construction,rhetorical,storyscope,style
    sd = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[proper] loaded (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    model.eval()

    report: dict = {"ckpt": args.ckpt, "device": device}

    # ---- 1. val registers, capped ----
    regs = sorted(set(HUMAN_REGS + AI_REGS))
    rows = load_rows_minimal(args.spans_parquet, regs, args.cap)
    # derive actual human/AI register sets from labels (works for v1 AND v2)
    seen_regs: dict[str, set[int]] = {}
    for _, y, rg in rows:
        seen_regs.setdefault(rg, set()).add(y)
    human_regs = sorted(r for r, ys in seen_regs.items() if ys == {0})
    ai_regs = sorted(r for r, ys in seen_regs.items() if ys == {1})
    mixed_regs = sorted(r for r, ys in seen_regs.items() if len(ys) > 1)
    print(f"[proper] human_regs={human_regs} ai_regs={ai_regs} mixed={mixed_regs}", flush=True)
    print(f"[proper] scoring {len(rows)} rows...", flush=True)
    t0 = time.time()
    scores, labels, rregs = score_rows(model, tok, rows, device, args.max_len)
    print(f"[proper] scored in {time.time()-t0:.1f}s", flush=True)

    report["per_register_stats"] = stats(scores, labels, rregs)

    # ---- 2. cross-register AUROC matrix (all human x all AI pairs) ----
    matrix = {}
    for h in human_regs:
        for a in ai_regs:
            hs = [(s, 0) for s, y, r in zip(scores, labels, rregs) if r == h]
            as_ = [(s, 1) for s, y, r in zip(scores, labels, rregs) if r == a]
            if not hs or not as_:
                continue
            ss = [x[0] for x in hs + as_]
            yy = [x[1] for x in hs + as_]
            matrix[f"{h}_vs_{a}"] = {"n": len(ss), "auroc": round(auroc(ss, yy), 4)}
    report["cross_register"] = matrix
    for k, v in matrix.items():
        print(f"[proper] {k:28s} n={v['n']:6d} auroc={v['auroc']}", flush=True)

    # ---- 3. coai strict ----
    coai = [(s, y) for s, y, r in zip(scores, labels, rregs) if r == "coai"]
    if coai:
        ss = [x[0] for x in coai]
        yy = [x[1] for x in coai]
        report["coai_strict"] = {"n": len(ss), "auroc": round(auroc(ss, yy), 4),
                                 "regex_floor": 0.8643}
        print(f"[proper] coai strict auroc={report['coai_strict']['auroc']} (floor 0.8643)", flush=True)

    # ---- 4. held-out jsonl files ----
    heldout = {}
    for path in HELD_OUT:
        p = Path(path)
        if not p.exists():
            heldout[p.stem] = {"error": "missing"}
            continue
        rows_h = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            label = d.get("pile", d.get("label"))
            if label is None or not d.get("text"):
                continue
            rows_h.append((d["text"], int(label), p.stem))
        if not rows_h:
            heldout[p.stem] = {"error": "no rows"}
            continue
        hs, hy, _ = score_rows(model, tok, rows_h, device, args.max_len)
        heldout[p.stem] = {"n": len(hs), "auroc": round(auroc(hs, hy), 4)}
        print(f"[proper] heldout {p.stem:16s} n={len(hs):4d} auroc={heldout[p.stem]['auroc']}", flush=True)
    report["heldout"] = heldout

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"[proper] wrote {out}/report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
