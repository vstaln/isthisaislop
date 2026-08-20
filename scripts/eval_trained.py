#!/usr/bin/env python3
"""Post-training eval + quantization check for the fine-tuned encoder.

Runs AFTER training (artifacts/lfm/model.pt exists). Two jobs:

1. Per-register evaluation on the held-out val slice — same logic as the end of
   fine_tune_lfm.main(), so you can re-check metrics without retraining, and
   compare against the CPU floor (artifacts/MANIFEST.json: coai AUC 0.8643).

2. Quantization probe: export the bundle to ONNX, INT8-dynamic quantize, report
   logit drift + CPU latency. The pre-training export gate (--probe) uses
   untrained heads; this uses the REAL trained weights so drift is meaningful.

Usage:
  uv run python scripts/eval_trained.py \
      --spans-parquet data/training/train_all.parquet \
      --ckpt artifacts/lfm/model.pt \
      --model LiquidAI/LFM2.5-Encoder-350M \
      --out artifacts/lfm_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from slopdet.lfm import load_encoder_body  # noqa: E402


def auroc(scores: list[float], labels: list[int]) -> float:
    """Area under ROC — same implementation as fine_tune_lfm."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = [i + 1 for i in order]
    sum_pos_ranks = sum(r for i, r in zip(order, ranks) if labels[i])
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def tpr_at_fpr(scores: list[float], labels: list[int], fpr: float = 0.01) -> tuple[float, float]:
    """Threshold at 1% FPR on humans, and the TPR the model gets there."""
    human = [(s, l) for s, l in zip(scores, labels) if l == 0]
    ai = [(s, l) for s, l in zip(scores, labels) if l == 1]
    if not human or not ai:
        return float("nan"), float("nan")
    human_scores = sorted(s[0] for s in human)
    k = max(0, min(len(human_scores) - 1, int(len(human_scores) * fpr)))
    thresh = human_scores[k]
    tpr = sum(1 for s, _ in ai if s >= thresh) / len(ai)
    return tpr, thresh


class DetectorBundle(nn.Module):
    """One body, two heads — mirror of the export gate so weights load directly."""

    def __init__(self, model_name: str, n_lanes: int):
        super().__init__()
        self.body = load_encoder_body(model_name)
        hidden = self.body.config.hidden_size
        self.doc = nn.Linear(hidden, 2)
        self.token = nn.Linear(hidden, n_lanes + 1)  # +1 for class 0 (no lane)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        states = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.doc(pooled), self.token(states)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans-parquet", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("artifacts/lfm/model.pt"))
    ap.add_argument("--model", default="LiquidAI/LFM2.5-Encoder-350M")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("artifacts/lfm_eval"))
    ap.add_argument("--quantize", action="store_true", help="also run the INT8 export probe")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}, ckpt={args.ckpt}")

    if not args.ckpt.exists():
        print(f"[eval] no checkpoint at {args.ckpt} — train first (Colab notebook).")
        return 1

    # ---- load rows + lanes (reuse fine_tune_lfm.load_rows) ----
    from fine_tune_lfm import load_rows
    import random

    rows = load_rows(None, args.spans_parquet, False)
    lanes = sorted({s["lane"] for r in rows for s in r["spans"]})
    # same per-(register,label) split as training
    random.Random(0).shuffle(rows)
    val_rows = []
    for register in {r["register"] for r in rows}:
        for label in (0, 1):
            group = [r for r in rows if r["register"] == register and r["label"] == label]
            take = max(1, int(len(group) * args.val_frac)) if group else 0
            val_rows += group[:take]
    print(f"[eval] {len(val_rows)} val rows, lanes={lanes}", flush=True)

    # ---- build model, load trained weights ----
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = DetectorBundle(args.model, len(lanes)).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    # state dict may be wrapped or bare
    if "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[eval] WARNING missing keys: {missing[:5]}")
    model.eval()

    # ---- score the val set ----
    scores, labels, regs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for r in val_rows:
            enc = tok(r["text"], truncation=True, max_length=args.max_len, return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            doc_logits, _ = model(ids, mask)
            prob = torch.softmax(doc_logits, -1)[0, 1].item()
            scores.append(prob)
            labels.append(r["label"])
            regs.append(r["register"])
    print(f"[eval] scored {len(val_rows)} in {time.time()-t0:.1f}s", flush=True)

    # ---- per-register metrics ----
    metrics: dict[str, dict] = {}
    for register in sorted(set(regs)):
        idx = [i for i, r in enumerate(regs) if r == register]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        tpr, thresh = tpr_at_fpr(s, y)
        metrics[register] = {
            "n": len(idx), "auroc": auroc(s, y),
            "tpr_at_1pct_fpr": tpr, "threshold": thresh,
        }
    tpr_all, thresh_all = tpr_at_fpr(scores, labels)
    metrics["all"] = {
        "n": len(labels), "auroc": auroc(scores, labels),
        "tpr_at_1pct_fpr": tpr_all, "threshold": thresh_all,
    }
    print(json.dumps(metrics, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[eval] wrote {args.out / 'eval.json'}")

    # ---- optional quantization probe with real weights ----
    if args.quantize:
        from export_onnx import _export, _latency  # reuse the gate's helpers
        import onnxruntime as ort

        onnx_path = args.out / "bundle.onnx"
        sample = (
            torch.randint(0, 60000, (1, args.max_len)),
            torch.ones(1, args.max_len, dtype=torch.long),
        )
        kind = _export(model, sample, onnx_path)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feeds = {
            "input_ids": sample[0].numpy(),
            "attention_mask": sample[1].numpy(),
        }
        # drift: fp32 (torch) vs int8 (onnx) doc logits on 50 val docs
        drift = []
        with torch.no_grad():
            for r in val_rows[:50]:
                enc = tok(r["text"], truncation=True, max_length=args.max_len, return_tensors="pt")
                ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
                doc_logits, _ = model(ids, mask)
                fp = doc_logits[0].cpu().numpy()
                q = sess.run(None, {"input_ids": ids.cpu().numpy(), "attention_mask": mask.cpu().numpy()})[0][0]
                drift.append(float(np.abs(fp - q).max()))
        probe = {
            "export": kind, "latency_ms": _latency(sess, feeds),
            "max_logit_drift_int8": float(np.max(drift)) if drift else None,
            "model": args.model,
        }
        print(json.dumps(probe, indent=2))
        (args.out / "export_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
        print(f"[eval] wrote {args.out / 'export_probe.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
