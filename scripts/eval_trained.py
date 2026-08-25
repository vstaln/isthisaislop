#!/usr/bin/env python3
"""Post-training evaluation — the one command to run after fine_tune_lfm.py.

    uv run python scripts/eval_trained.py --ckpt artifacts/lfm_v2/model.pt

Per-register metrics and the 1%-FPR calibration come out of a single pass over the
same pair-safe split the trainer used: the same corpus with the same --val-frac /
--cal-frac / --seed reproduces the same val and cal rows, so these numbers are
comparable to the training manifest rather than a second opinion from a slightly
different slice.

What it writes to --out:
  eval.json            per-register AUROC (gated + raw), TPR@1%FPR, n, threshold
  calibration.json     the operating threshold per register, fitted on cal only
  cross_register.json  human-register x AI-register AUROC matrix
  export_probe.json    INT8 drift and CPU latency, with --quantize

Read the cross-register matrix before believing the headline. A model that learnt
register or era rather than provenance scores near 1.0 on every cross pair while
sitting at chance inside a register that holds both classes — that is how v1
failed (docs/HANDOFF.md, V1 POSTMORTEM).

--final-report additionally scores the held-out vault (eval/labels/*.jsonl plus the
generator-disjoint HF holdout parquets). Those exist to be reported once, never to
select a checkpoint against.
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

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from slopdet import corpus  # noqa: E402
from slopdet.detector import from_checkpoint  # noqa: E402
from slopdet.metrics import (  # noqa: E402
    auroc,
    cross_register_auroc,
    per_register_metrics,
    register_thresholds,
)
from slopdet.pairs import split_rows  # noqa: E402

VAULT_JSONL = ("eval/labels/laguna.jsonl", "eval/labels/local.jsonl",
               "eval/labels/deepseek_eval.jsonl")


@torch.no_grad()
def score(model, tok, texts: list[str], device, max_len: int, batch_size: int) -> list[float]:
    out: list[float] = []
    for start in range(0, len(texts), batch_size):
        enc = tok(texts[start:start + batch_size], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        doc_logits, _ = model(enc["input_ids"], enc["attention_mask"])
        out.extend(torch.softmax(doc_logits.float(), -1)[:, 1].cpu().tolist())
    return out


def read_jsonl(path: Path) -> list[tuple[str, int]]:
    """(text, label) pairs from a hand-labeled eval file; `label` or `pile`, either way."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        label = rec.get("label", rec.get("pile"))
        if label is None or not rec.get("text"):
            continue
        rows.append((rec["text"], int(label)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("artifacts/lfm/model.pt"))
    ap.add_argument("--corpus", "--spans-parquet", dest="corpus", type=Path,
                    default=Path("data/v2") / corpus.V2_TRAIN_FILE,
                    help="the corpus the checkpoint was trained on")
    ap.add_argument("--hf-file", default=corpus.V2_TRAIN_FILE)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-Encoder-350M")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--cal-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="must match the training run")
    ap.add_argument("--out", type=Path, default=Path("artifacts/lfm_eval"))
    ap.add_argument("--final-report", action="store_true",
                    help="also score the held-out vault — report only, never select on it")
    ap.add_argument("--quantize", action="store_true", help="also run the INT8 export probe")
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"[eval] no checkpoint at {args.ckpt} — run scripts/fine_tune_lfm.py first.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device} ckpt={args.ckpt}", flush=True)

    rows = corpus.load_rows(corpus.resolve(args.corpus, args.hf_file))
    _, val_rows, cal_rows = split_rows(rows, args.val_frac, args.cal_frac, args.seed)
    print(f"[eval] {len(val_rows)} val / {len(cal_rows)} cal rows over "
          f"{len({r['register'] for r in val_rows})} registers", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = from_checkpoint(args.ckpt, args.model, device)
    print(f"[eval] checkpoint carries {model.n_lanes} lanes", flush=True)

    started = time.time()
    val_scores = score(model, tok, [r["text"] for r in val_rows], device, args.max_len, args.batch_size)
    cal_scores = score(model, tok, [r["text"] for r in cal_rows], device, args.max_len, args.batch_size)
    print(f"[eval] scored {len(val_scores) + len(cal_scores)} docs in {time.time() - started:.1f}s",
          flush=True)

    val_labels = [r["label"] for r in val_rows]
    val_regs = [r["register"] for r in val_rows]
    thresholds = register_thresholds(cal_scores, [r["label"] for r in cal_rows],
                                     [r["register"] for r in cal_rows])
    metrics = per_register_metrics(val_scores, val_labels, val_regs, thresholds)
    matrix = cross_register_auroc(val_scores, val_labels, val_regs)

    if args.final_report:
        print("[eval] VAULT: final-reporting slices — selecting a checkpoint against "
              "these invalidates them", flush=True)
        for name in VAULT_JSONL:
            path = ROOT / name
            if not path.exists():
                metrics[f"vault:{Path(name).stem}"] = {"error": "missing"}
                continue
            pairs = read_jsonl(path)
            scores = score(model, tok, [t for t, _ in pairs], device, args.max_len, args.batch_size)
            metrics[f"vault:{Path(name).stem}"] = {
                "n": len(pairs), "auroc": auroc(scores, [y for _, y in pairs])}
        for hf_file in corpus.V2_HOLDOUT_FILES:
            holdout = corpus.load_rows(corpus.resolve(Path("data/v2") / hf_file, hf_file))
            scores = score(model, tok, [r["text"] for r in holdout], device,
                           args.max_len, args.batch_size)
            metrics[f"vault:{Path(hf_file).stem}"] = {
                "n": len(holdout), "auroc": auroc(scores, [r["label"] for r in holdout])}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out / "cross_register.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (args.out / "calibration.json").write_text(json.dumps(
        {"fpr": 0.01,
         "per_register": {k: v["threshold"] for k, v in metrics.items() if "threshold" in v},
         "threshold_source": "cal" if cal_scores else "val"}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if matrix:
        print("[eval] cross-register AUROC (near 1.0 everywhere = register shortcut):")
        for name, entry in sorted(matrix.items()):
            print(f"  {name:44s} n={entry['n']:6d} auroc={entry['auroc']:.4f}")
    print(f"[eval] wrote {args.out}/eval.json, cross_register.json, calibration.json")

    if args.quantize:
        import numpy as np
        import onnxruntime as ort

        from export_onnx import export_graph, latency_ms

        onnx_path = args.out / "bundle.onnx"
        sample = (torch.randint(0, 60000, (1, args.max_len)),
                  torch.ones(1, args.max_len, dtype=torch.long))
        kind = export_graph(model, sample, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        drift = []
        with torch.no_grad():
            for row in val_rows[:50]:
                enc = tok(row["text"], truncation=True, max_length=args.max_len, return_tensors="pt")
                ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
                fp32 = model(ids, mask)[0][0].cpu().numpy()
                int8 = session.run(None, {"input_ids": ids.cpu().numpy(),
                                          "attention_mask": mask.cpu().numpy()})[0][0]
                drift.append(float(np.abs(fp32 - int8).max()))
        probe = {"export": kind, "model": args.model,
                 "latency_ms": latency_ms(session, {"input_ids": sample[0].numpy(),
                                                    "attention_mask": sample[1].numpy()}),
                 "max_logit_drift_int8": max(drift) if drift else None}
        (args.out / "export_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
        print(json.dumps(probe, indent=2))
        print(f"[eval] wrote {args.out}/export_probe.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
