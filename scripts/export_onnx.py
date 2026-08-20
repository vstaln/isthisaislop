"""Export gate: can the chosen backbone reach the device at all?

Exports the encoder body plus our doc/token heads to ONNX, quantizes to INT8 dynamic, then reports
logit drift and CPU latency. Runs on CPU in minutes with untrained heads — the point is to retire the
deployment risk *before* spending Colab time on training.

    uv run python scripts/export_onnx.py --arch encoder --probe

Writes artifacts/export_probe.json and exits nonzero if export, quantization or the drift check fails.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

from slopdet.lfm import load_encoder_body

DEFAULT_MODEL = "LiquidAI/LFM2.5-Encoder-350M"
N_LANES = 8


class DetectorBundle(nn.Module):
    """One body, two heads: pooled doc score + per-token lane logits."""

    def __init__(self, model_name: str, n_lanes: int = N_LANES):
        super().__init__()
        self.body = load_encoder_body(model_name)
        hidden = self.body.config.hidden_size
        self.doc = nn.Linear(hidden, 2)
        self.token = nn.Linear(hidden, n_lanes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        states = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.doc(pooled), self.token(states)


def _export(model: nn.Module, sample: tuple[torch.Tensor, torch.Tensor], path: Path) -> str:
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "doc_logits": {0: "batch"},
        "token_logits": {0: "batch", 1: "seq"},
    }
    kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["doc_logits", "token_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    try:
        torch.onnx.export(model, sample, str(path), dynamo=True, **kwargs)
        return "dynamo"
    except Exception as exc:  # noqa: BLE001 - the fallback is the whole point
        print(f"[probe] dynamo export failed ({type(exc).__name__}: {exc}); retrying legacy")
        torch.onnx.export(model, sample, str(path), dynamo=False, **kwargs)
        return "torchscript"


def _latency(session, feeds: dict[str, np.ndarray], runs: int = 10) -> float:
    session.run(None, feeds)
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, feeds)
    return (time.perf_counter() - start) / runs * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["encoder"], default="encoder")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--probe", action="store_true", help="untrained heads; measure only")
    ap.add_argument("--out", type=Path, default=Path("artifacts/onnx"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    args.out.mkdir(parents=True, exist_ok=True)
    fp32_path = args.out / "detector.onnx"
    int8_path = args.out / "detector.int8.onnx"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = DetectorBundle(args.model).eval()

    text = "Here's the thing: we leverage robust pipelines to unlock synergies. " * 12
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    sample = (enc["input_ids"], enc["attention_mask"])

    with torch.no_grad():
        torch_doc, torch_token = model(*sample)

    exporter = _export(model, sample, fp32_path)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    feeds = {k: v.numpy() for k, v in enc.items() if k in {"input_ids", "attention_mask"}}
    results = {}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        doc, token = sess.run(None, feeds)
        results[name] = {
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "doc_drift": float(np.abs(doc - torch_doc.numpy()).max()),
            "token_drift": float(np.abs(token - torch_token.numpy()).max()),
            "latency_ms_512": round(_latency(sess, feeds), 1),
        }
        long_enc = tok(text * 3, return_tensors="np", truncation=True, max_length=1024, padding="max_length")
        long_feeds = {k: long_enc[k] for k in ("input_ids", "attention_mask")}
        results[name]["latency_ms_1024"] = round(_latency(sess, long_feeds), 1)

    report = {
        "model": args.model,
        "exporter": exporter,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "threads": args.threads,
        "untrained_heads": bool(args.probe),
        **results,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/export_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if results["fp32"]["doc_drift"] > 1e-3:
        print("[probe] FAIL: fp32 ONNX disagrees with torch beyond 1e-3")
        return 1
    print("[probe] PASS: export + INT8 quantization viable on this backbone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
