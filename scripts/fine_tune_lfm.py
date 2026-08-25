"""Fine-tune an LFM2 backbone into the ITAIS detector.

Two paths, one script. The encoder path is the recommended default (see docs/HANDOFF.md §3):

    # recommended: bidirectional encoder, doc head + per-token lane head, one body
    uv run python scripts/fine_tune_lfm.py --arch encoder --model LiquidAI/LFM2.5-Encoder-230M

    # rev-1 alternative: causal decoder + LoRA sequence classifier, doc verdict only
    uv run python scripts/fine_tune_lfm.py --arch decoder --model LiquidAI/LFM2.5-1.2B --max-len 2048

    # no corpus needed: synthetic data, 3 steps, proves the graph/loss/export path works
    uv run python scripts/fine_tune_lfm.py --arch encoder --smoke

Outputs a bundle under --out: weights, tokenizer, calibration.json (threshold per register at 1% FPR)
and manifest.json (args, git sha, seed, per-slice metrics, trained_on / never_trained_on).

Never emits or stores a "% AI" number: the doc head is a pile-resemblance score, calibrated against a
named human reference slice. See docs/HANDOFF.md §1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random
import subprocess
import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Ensure `slopdet` (src/) is importable no matter the CWD. Colab runs this as
# `python scripts/fine_tune_lfm.py` from a Drive dir; uv runs it from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from slopdet.labels import parse_label
from slopdet.lfm import load_encoder_body

DOC_LABELS = {"human": 0, "ai": 1}


@dataclass
class Config:
    arch: str = "encoder"
    model: str = "LiquidAI/LFM2.5-Encoder-350M"
    max_len: int = 512
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 2e-5
    epochs: int = 1
    seed: int = 0
    token_loss_weight: float = 0.5
    precision: str = "fp16"
    ckpt_every: int = 500


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- data


def load_rows(doc_parquet: Path | None, spans_parquet: Path | None, smoke: bool) -> list[dict]:
    """Rows are dicts: text, label (0/1), spans (list of {lane,start,end}), register."""
    if smoke:
        slop = "Here's the thing: we leverage robust pipelines to unlock synergies. "
        human = "Thursday mornings at the clinic were empty, so I counted 41 chairs. "
        rows = []
        for i in range(24):
            rows.append({"text": slop * 3, "label": 1, "register": "smoke",
                         "spans": [{"lane": "glue", "start": 25, "end": 33}]})
            rows.append({"text": human * 3, "label": 0, "register": "smoke", "spans": []})
        return rows

    import pandas as pd

    def _coerce_spans(spans) -> list[dict]:
        """Return the span dicts that carry a lane, tolerating None/str/list/ndarray."""
        if spans is None:
            return []
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except json.JSONDecodeError:
                return []
        if isinstance(spans, (list, tuple)):
            return [s for s in spans if isinstance(s, dict) and s.get("lane")]
        # numpy array / Arrow list — coerce safely
        try:
            return [s for s in spans if isinstance(s, dict) and s.get("lane")]
        except TypeError:
            return []

    def _read_chunked(path: Path, cols: list[str], chunk: int = 10_000):
        """Stream a parquet in slices so peak RAM stays bounded on Colab."""
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunk, columns=cols):
            yield batch.to_pandas()

    if spans_parquet and spans_parquet.exists():
        import pyarrow.parquet as pq
        # logical columns (spans is a single list column, not lane/start/end leaves)
        schema_names = set(pq.read_schema(spans_parquet).names)
        cols = [c for c in ("text", "label", "pile", "register", "spans", "split_hint")
                if c in schema_names]
        if "register" not in cols:
            raise SystemExit(f"{spans_parquet}: missing 'register' column — rebuild with build_training_parquet.py (m3 fix)")
        if "text" not in cols:
            raise SystemExit(f"{spans_parquet}: missing 'text' column")
        rows = []
        for chunk_df in _read_chunked(spans_parquet, cols):
            for rec in chunk_df.to_dict("records"):
                reg = rec.get("register")
                if not reg:
                    raise SystemExit(f"row missing register: {rec}")
                # Require explicit label/pile — parse_label raises on garbage; default 0 only if both absent (should not happen in train_all)
                if rec.get("label") is None and rec.get("pile") is None:
                    raise SystemExit(f"row missing label/pile: register={reg}")
                rows.append({
                    "text": rec["text"],
                    "label": parse_label(rec, default=0),
                    "register": reg,
                    "spans": _coerce_spans(rec.get("spans")),
                    # pair key for contrastive training (absent in v1 parquets)
                    "split_hint": rec.get("split_hint") or "",
                })
        return rows

    if not doc_parquet or not doc_parquet.exists():
        raise SystemExit(
            f"no corpus at {doc_parquet} / {spans_parquet}. Rebuild it with the commands in "
            "docs/HANDOFF.md §4, or pass --smoke to validate the code path without data."
        )
    rows = []
    for chunk_df in _read_chunked(doc_parquet, ["text", "label", "register"]):
        for rec in chunk_df.to_dict("records"):
            rows.append({"text": rec["text"], "label": int(rec["label"]),
                         "register": rec.get("register", "coai"), "spans": []})
    return rows


class SlopDataset(Dataset):
    def __init__(self, rows: list[dict], tok, max_len: int, lanes: list[str]):
        self.rows, self.tok, self.max_len = rows, tok, max_len
        self.lane_ids = {lane: i + 1 for i, lane in enumerate(lanes)}  # 0 = no lane

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        enc = self.tok(row["text"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        token_labels = [0] * len(offsets)
        for span in row["spans"]:
            lane_id = self.lane_ids.get(span["lane"])
            if not lane_id:
                continue
            for pos, (start, end) in enumerate(offsets):
                if end > start and start >= span["start"] and end <= span["end"]:
                    token_labels[pos] = lane_id
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["doc_label"] = torch.tensor(row["label"])
        item["token_labels"] = torch.tensor(token_labels)
        return item


# ---------------------------------------------------------------- model


class EncoderDetector(nn.Module):
    """LFM2 bidirectional body, mean-pooled doc head + per-token lane head."""

    def __init__(self, model_name: str, n_lanes: int):
        super().__init__()
        self.body = load_encoder_body(model_name)
        hidden = self.body.config.hidden_size
        self.doc = nn.Linear(hidden, 2)
        self.token = nn.Linear(hidden, n_lanes + 1)

    def _encode(self, input_ids, attention_mask):
        states = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return states, pooled

    def forward(self, input_ids, attention_mask):
        states, pooled = self._encode(input_ids, attention_mask)
        return self.doc(pooled), self.token(states)

    def forward_doc_embed(self, input_ids, attention_mask):
        """Penultimate (pooled hidden) embedding + doc logits for contrastive loss."""
        _, pooled = self._encode(input_ids, attention_mask)
        return pooled, self.doc(pooled)


def build_decoder(model_name: str):
    """Rev-1 path: causal LM with a 2-class head over the last token, LoRA-adapted."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, trust_remote_code=True)
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    return get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="SEQ_CLS", target_modules="all-linear"))


# ---------------------------------------------------------------- metrics


def auroc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum, rank = 0.0, 0
    while rank < len(pairs):
        tied = [i for i in range(rank, len(pairs)) if pairs[i][0] == pairs[rank][0]]
        avg_rank = sum(i + 1 for i in tied) / len(tied)
        rank_sum += sum(avg_rank for i in tied if pairs[i][1] == 1)
        rank += len(tied)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def tpr_at_fpr(scores: list[float], labels: list[int], fpr: float = 0.01) -> tuple[float, float]:
    from slopdet.calibrate import threshold_at_fpr

    human = [s for s, y in zip(scores, labels) if y == 0]
    if not human:
        return float("nan"), float("nan")
    threshold = threshold_at_fpr(human, fpr)
    ai = [s for s, y in zip(scores, labels) if y == 1]
    tpr = sum(s > threshold for s in ai) / len(ai) if ai else float("nan")
    return tpr, threshold


def _gated_auroc(scores: list[float], labels: list[int], min_n: int = 200, min_pos: int = 20) -> float:
    """Return AUROC or nan if too few samples to be meaningful (M4 gate)."""
    if len(labels) < min_n or sum(labels) < min_pos or sum(1 for y in labels if y == 0) < min_pos:
        return float("nan")
    return auroc(scores, labels)


# ---------------------------------------------------------------- train


def evaluate(model, loader, device, arch: str) -> tuple[list[float], list[int]]:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            doc_logits = model(ids, mask)[0] if arch == "encoder" else model(
                input_ids=ids, attention_mask=mask).logits
            scores += torch.softmax(doc_logits.float(), -1)[:, 1].cpu().tolist()
            labels += batch["doc_label"].tolist()
    model.train()
    return scores, labels


class _MultiOptimizer:
    """Step two optimizers together so one scheduler covers both groups."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.param_groups = a.param_groups + b.param_groups
        self.defaults = {}

    def step(self, closure=None):
        self.a.step()
        self.b.step()

    def zero_grad(self, set_to_none=True):
        self.a.zero_grad(set_to_none)
        self.b.zero_grad(set_to_none)

    def state_dict(self):
        return {"a": self.a.state_dict(), "b": self.b.state_dict()}

    def load_state_dict(self, sd):
        self.a.load_state_dict(sd["a"])
        self.b.load_state_dict(sd["b"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["encoder", "decoder"], default="encoder")
    ap.add_argument("--model", default=Config.model)
    ap.add_argument("--doc-parquet", type=Path, default=Path("data/coai_train.parquet"))
    ap.add_argument("--spans-parquet", type=Path, default=Path("data/training/spans_coai_train.parquet"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/lfm"))
    ap.add_argument("--max-len", type=int, default=Config.max_len)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--grad-accum", type=int, default=Config.grad_accum)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default=Config.precision)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                    help="muon = Moonshot-style hybrid: orthogonalized updates on matmul "
                         "weights, AdamW on embeddings/conv/norm/heads")
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="Muon group learning rate (AdamW groups keep --lr)")
    ap.add_argument("--ckpt-every", type=int, default=Config.ckpt_every)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--cal-frac", type=float, default=0.05)
    ap.add_argument("--warmup-frac", type=float, default=0.1, help="fraction of steps for linear warmup (0=off)")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, 3 steps, no corpus needed")
    ap.add_argument("--contrastive", type=float, default=0.0,
                    help="InfoNCE weight on same-prompt pairs (0=off; also gates per-step probability when <1)")
    args = ap.parse_args()

    cfg = Config(arch=args.arch, model=args.model, max_len=args.max_len, batch_size=args.batch_size,
                 grad_accum=args.grad_accum, lr=args.lr, epochs=args.epochs, seed=args.seed,
                 precision=args.precision, ckpt_every=args.ckpt_every)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.precision == "fp16" and device.type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        print("[warn] Turing-class GPU: no bf16 and no flash-attn 2. fp16 + NaN preflight it is.")

    rows = load_rows(args.doc_parquet, args.spans_parquet, args.smoke)
    lanes = sorted({s["lane"] for r in rows for s in r["spans"]})
    # Schema check: register allowlist + label 0/1 (labels.py already raises on pile/label mismatch)
    from slopdet.calibrate import ALLOWED_REGISTERS, register_allowed
    bad_regs = {r["register"] for r in rows
                if not register_allowed(r["register"])}
    if bad_regs:
        raise SystemExit(f"unknown register(s) {bad_regs} — fix ALLOWED_REGISTERS or data build")
    bad_labels = {r["label"] for r in rows} - {0, 1}
    if bad_labels:
        raise SystemExit(f"bad label values {bad_labels}")
    random.Random(args.seed).shuffle(rows)
    # Split per (register,label) into train / val / cal — cal distinct from val so thresholds not circular (M2 fix).
    val_rows, cal_rows, train_rows = [], [], []
    for register in sorted({r["register"] for r in rows}):
        for label in (0, 1):
            group = [r for r in rows if r["register"] == register and r["label"] == label]
            n_val = max(1, int(len(group) * args.val_frac)) if group else 0
            n_cal = max(1, int(len(group) * args.cal_frac)) if group else 0
            # If group tiny, don't consume all rows for val+cal
            if len(group) and n_val + n_cal >= len(group):
                n_val = max(1, len(group) // 10) if len(group) >= 10 else 1
                n_cal = 1 if len(group) >= 3 else 0
                if n_val + n_cal >= len(group):
                    n_cal = 0
            val_rows += group[:n_val]
            cal_rows += group[n_val:n_val + n_cal]
            train_rows += group[n_val + n_cal:]
    random.Random(args.seed).shuffle(train_rows)
    print(f"[data] {len(train_rows)} train / {len(val_rows)} val / {len(cal_rows)} cal · {len(lanes)} lanes · registers {sorted({r['register'] for r in rows})}")

    ctr_pairs: list[tuple[int, int]] = []
    ctr_rng = random.Random(args.seed + 1)
    if args.contrastive > 0.0:
        from slopdet.pairs import build_pairs
        ctr_pairs = build_pairs(train_rows)
        print(f"[ctr] pairs={len(ctr_pairs)} usable", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    train_ds = SlopDataset(train_rows, tok, cfg.max_len, lanes)
    val_ds = SlopDataset(val_rows, tok, cfg.max_len, lanes)
    cal_ds = SlopDataset(cal_rows, tok, cfg.max_len, lanes) if cal_rows else None
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
                          num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=2, pin_memory=True)
    cal_dl = DataLoader(cal_ds, batch_size=cfg.batch_size, num_workers=2, pin_memory=True) if cal_ds else None

    model = (EncoderDetector(cfg.model, len(lanes)) if cfg.arch == "encoder"
             else build_decoder(cfg.model)).to(device)

    # ---- optimizer: adamw (default) or moonshot-style hybrid muon ----
    # Muon (Jordan et al. 2024; Moonshot AI 2502.16982): orthogonalized
    # momentum updates on 2D matmul weights ONLY; embeddings, norms, heads,
    # and conv weights stay on AdamW. Newton-Schulz runs in bf16 on GPU.
    if args.optimizer == "muon":
        import torch.nn.functional as F

        @torch.no_grad()
        def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
            a, b, c = (3.4445, -4.7750, 2.0315)
            X = G.bfloat16()
            transposed = False
            if X.size(0) > X.size(1):
                X = X.T
                transposed = True
            X = X / (X.norm() + 1e-7)
            for _ in range(steps):
                A = X @ X.T
                B = b * A + c * A @ A
                X = a * X + B @ X
            if transposed:
                X = X.T
            return X

        class Muon(torch.optim.Optimizer):
            """DeepSeek-V4 spec (arxiv 2606.19348 §2.4): momentum 0.95,
            weight decay 0.1, Newton-Schulz orthogonalization, update-matrix
            RMS rescaled to 0.18 so the AdamW learning rate is reused directly."""

            UPDATE_RMS = 0.18

            def __init__(self, params, lr, momentum=0.95, weight_decay=0.1):
                super().__init__(params, dict(lr=lr, momentum=momentum,
                                              weight_decay=weight_decay))

            @torch.no_grad()
            def step(self):
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        g = p.grad
                        st = self.state[p]
                        if "m" not in st:
                            st["m"] = torch.zeros_like(g)
                        m = st["m"]
                        m.lerp_(g, 1 - group["momentum"])
                        u = g.lerp_(m, group["momentum"])
                        flat = u.flatten(1) if u.ndim > 2 else u
                        if flat.ndim < 2:  # 1D param slipped in — momentum fallback
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            p.add_(u, alpha=-group["lr"])
                            continue
                        o = _zeropower_via_newtonschulz(flat).to(p.dtype)
                        # RMS-rescale to 0.18 -> reuse AdamW lr (DeepSeek-V4)
                        rms = o.float().pow(2).mean().sqrt().clamp_min(1e-12)
                        o = o * (self.UPDATE_RMS / rms)
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(o.reshape(p.shape), alpha=-group["lr"])

        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_hidden_matmul = (p.ndim >= 2
                                and name.split(".")[0] not in {"doc", "token"}
                                and not any(k in name for k in
                                            ("embed", "head", "norm", "classifier",
                                             "doc_head", "lane_head", "conv")))
            (muon_params if is_hidden_matmul else adamw_params).append(p)
        n_mu = sum(p.numel() for p in muon_params)
        n_ad = sum(p.numel() for p in adamw_params)
        print(f"[opt] muon(dsv4) on {n_mu/1e6:.1f}M params (matmul weights), "
              f"adamw on {n_ad/1e6:.1f}M (embed/conv/norm/heads)", flush=True)
        opt = Muon(muon_params, lr=cfg.lr, weight_decay=0.1)
        opt_adamw = torch.optim.AdamW(adamw_params, lr=cfg.lr, weight_decay=0.01)
        opt = _MultiOptimizer(opt, opt_adamw)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    # Warmup + cosine decay (M1 fix) — constant LR is a coin flip over 21k steps.
    total_opt_steps = max(1, len(train_dl) // cfg.grad_accum) * cfg.epochs
    warmup_steps = int(total_opt_steps * args.warmup_frac) if not args.smoke else 0
    scheduler = None
    if warmup_steps > 0 and total_opt_steps > 1:
        try:
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_opt_steps)
            print(f"[sched] cosine warmup {warmup_steps}/{total_opt_steps}", flush=True)
        except Exception:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warm = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(opt, T_max=max(1, total_opt_steps - warmup_steps))
            scheduler = SequentialLR(opt, [warm, cosine], milestones=[warmup_steps])
            print(f"[sched] fallback warmup {warmup_steps}/{total_opt_steps}", flush=True)
    use_amp = cfg.precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    doc_loss_fn = nn.CrossEntropyLoss()
    # K4 fix: class-balanced token loss. >99% of tokens are "no lane" (class 0),
    # so an unweighted token CE collapses to "no lane everywhere". Weight each
    # lane class inversely to its frequency so the head actually learns spans.
    lane_ids = {lane: i + 1 for i, lane in enumerate(lanes)}  # 0 = no lane
    token_weights = torch.ones(len(lanes) + 1, device=device)  # +1 for class 0
    lane_counts = torch.zeros(len(lanes) + 1, device=device)
    for r in train_rows:  # train only — no holdout span statistics in the loss
        for s in r["spans"]:
            lid = lane_ids.get(s["lane"], 0)
            lane_counts[lid] += max(0, (s["end"] - s["start"]))
    total = lane_counts.sum().clamp(min=1)
    # inverse-frequency, capped so rare lanes don't explode
    token_weights[1:] = (total / lane_counts[1:].clamp(min=1)).clamp(max=50)
    print(f"[data] token class weights: {token_weights.tolist()}", flush=True)
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=token_weights)
    try:
        del rows
    except NameError:
        pass
    gc.collect()

    total_steps = max(1, len(train_dl) // cfg.grad_accum) * cfg.epochs
    max_steps = 3 if args.smoke else total_steps * cfg.grad_accum
    args.out.mkdir(parents=True, exist_ok=True)
    _resume_ckpt = args.out / "checkpoint.pt"
    _resume_step = 0
    _resume_best = float("-inf")
    if _resume_ckpt.exists() and not args.smoke:
        try:
            _ck = torch.load(str(_resume_ckpt), map_location="cpu")
            if isinstance(_ck, dict) and "model" in _ck:
                model.load_state_dict(_ck["model"])
                _resume_step = int(_ck.get("step", 0))
                # restore full training state so resume == uninterrupted run
                try:
                    if "opt" in _ck:
                        opt.load_state_dict(_ck["opt"])
                    if scheduler is not None and _ck.get("scheduler"):
                        scheduler.load_state_dict(_ck["scheduler"])
                    if _ck.get("scaler"):
                        scaler.load_state_dict(_ck["scaler"])
                    if _ck.get("rng") is not None:
                        torch.set_rng_state(_ck["rng"])
                    print("[resume] restored optimizer/scheduler/scaler/rng state", flush=True)
                except Exception as _re:
                    print(f"[resume] WARNING: training-state restore failed ({_re}) — continuing with fresh state", flush=True)
                print(f"[resume] loaded checkpoint step {_resume_step} from {_resume_ckpt}", flush=True)
            del _ck; gc.collect()
        except Exception as e:
            print(f"[resume] failed {e}", flush=True)
    step, started = _resume_step, time.time()
    best_auroc = _resume_best
    best_state = None
    try:
        _bp = args.out / "best.pt"
        if _bp.exists():
            _b = torch.load(str(_bp), map_location="cpu")
            best_auroc = float(_b.get("auroc", best_auroc))
            del _b; gc.collect()
    except Exception:
        pass

    for epoch in range(cfg.epochs):
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            doc_y = batch["doc_label"].to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if cfg.arch == "encoder":
                    doc_logits, token_logits = model(ids, mask)
                    token_y = batch["token_labels"].to(device).masked_fill(mask == 0, -100)
                    loss = doc_loss_fn(doc_logits, doc_y) + cfg.token_loss_weight * token_loss_fn(
                        token_logits.reshape(-1, token_logits.size(-1)), token_y.reshape(-1))
                else:
                    loss = doc_loss_fn(model(input_ids=ids, attention_mask=mask).logits, doc_y)
                # Same-prompt InfoNCE: with prob min(1.0, contrastive), pull paired
                # human/AI embeddings apart via diag-correct cross-entropy over the
                # N x N cosine sim matrix. Falls back to normal loss silently when
                # <2 pairs exist or flag is off.
                if ctr_pairs and cfg.arch == "encoder" and ctr_rng.random() < min(1.0, args.contrastive):
                    n = min(len(ctr_pairs), max(cfg.batch_size // 2, 2))
                    if n >= 2:
                        samp = ctr_rng.sample(ctr_pairs, n)
                        h_enc = tok([train_rows[i]["text"] for i, _ in samp], padding=True, truncation=True,
                                    max_length=cfg.max_len, return_tensors="pt").to(device)
                        a_enc = tok([train_rows[j]["text"] for _, j in samp], padding=True, truncation=True,
                                    max_length=cfg.max_len, return_tensors="pt").to(device)
                        h_emb, _ = model.forward_doc_embed(h_enc["input_ids"], h_enc["attention_mask"])
                        a_emb, _ = model.forward_doc_embed(a_enc["input_ids"], a_enc["attention_mask"])
                        sim = nn.functional.normalize(h_emb.float(), dim=-1) @ \
                              nn.functional.normalize(a_emb.float(), dim=-1).t() / 0.07
                        ctr_loss = nn.functional.cross_entropy(sim, torch.arange(n, device=device)) * args.contrastive
                        loss = loss + ctr_loss
            if not torch.isfinite(loss):
                raise SystemExit(
                    "[abort] non-finite loss. On Turing (T4) fp16 is the usual cause: rerun with "
                    "--precision fp32, or lower --lr. Do not 'fix' this by filtering the log."
                )
            scaler.scale(loss / cfg.grad_accum).backward()
            step += 1
            if step % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                opt.zero_grad(set_to_none=True)
            if step % 50 == 0 or args.smoke:
                print(f"[train] epoch {epoch} step {step}/{max_steps} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
            if cfg.ckpt_every and step % cfg.ckpt_every == 0:
                _tmp_ckpt = args.out / "checkpoint.tmp"
                torch.save({"step": step, "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "scaler": scaler.state_dict(),
                            "rng": torch.get_rng_state()}, _tmp_ckpt)
                import os as _os; _os.replace(_tmp_ckpt, args.out / "checkpoint.pt")
                # auto HF backup every ckpt (VM -> HF durable, survives prune)
                try:
                    _hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                    if _hf and step % 1000 == 0:  # push every 1000 to avoid rate limit (1.4G)
                        from huggingface_hub import upload_file
                        upload_file(path_or_fileobj=str(args.out / "checkpoint.pt"), path_in_repo="checkpoint.pt", repo_id="vstalingrady/lfm-ckpt", repo_type="model", token=_hf)
                        print(f"[hf] pushed checkpoint step {step}", flush=True)
                except Exception as _e:
                    print(f"[hf] push skip {_e}", flush=True)
                # Lightweight val check for best-model selection (M1 fix) — no grad, on val set.
                if not args.smoke and len(val_rows) >= 200 and step % 2000 == 0:
                    v_scores, v_labels = evaluate(model, val_dl, device, cfg.arch)
                    v_auroc = _gated_auroc(v_scores, v_labels)
                    if v_auroc == v_auroc and v_auroc > best_auroc:  # not nan
                        best_auroc = v_auroc
                        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        torch.save({"step": step, "model": best_state, "auroc": best_auroc}, args.out / "best.pt")
                        print(f"[ckpt] new best val AUROC {best_auroc:.4f} at step {step}", flush=True)
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    # Final eval: thresholds from cal set (circular fix M2), metrics from val set with gating (M4).
    scores, labels = evaluate(model, val_dl, device, cfg.arch)
    cal_scores, cal_labels, cal_regs = [], [], []
    if cal_dl is not None:
        cal_scores, cal_labels = evaluate(model, cal_dl, device, cfg.arch)
        cal_regs = [r["register"] for r in cal_rows]
    registers = [r["register"] for r in val_rows]
    # Build per-register thresholds from cal set if available, else fallback to val human scores per register.
    cal_thresh: dict[str, float] = {}
    if cal_scores:
        for reg in sorted(set(cal_regs)):
            idx = [i for i, r in enumerate(cal_regs) if r == reg]
            s = [cal_scores[i] for i in idx]
            y = [cal_labels[i] for i in idx]
            _, thr = tpr_at_fpr(s, y)
            cal_thresh[reg] = thr
        # global fallback
        _, thr_all = tpr_at_fpr(cal_scores, cal_labels)
        cal_thresh["all"] = thr_all
    metrics: dict[str, dict] = {}
    for register in sorted(set(registers)):
        idx = [i for i, r in enumerate(registers) if r == register]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        # Use cal threshold if available, else val-derived (smoke or tiny cal)
        thr = cal_thresh.get(register, tpr_at_fpr(s, y)[1])
        tpr = sum(1 for sc, lb in zip(s, y) if lb == 1 and sc > thr) / max(1, sum(1 for lb in y if lb == 1))
        if sum(1 for lb in y if lb == 1) == 0 or sum(1 for lb in y if lb == 0) == 0:
            tpr = float("nan")
        gated = _gated_auroc(s, y)
        raw = auroc(s, y)
        metrics[register] = {"n": len(idx), "auroc": gated, "auroc_raw": raw,
                             "tpr_at_1pct_fpr": tpr, "threshold": thr,
                             "threshold_source": "cal" if register in cal_thresh else "val"}
    # all
    all_thr = cal_thresh.get("all", tpr_at_fpr(scores, labels)[1])
    all_tpr = sum(1 for sc, lb in zip(scores, labels) if lb == 1 and sc > all_thr) / max(1, sum(1 for lb in labels if lb == 1))
    metrics["all"] = {"n": len(labels), "auroc": _gated_auroc(scores, labels), "auroc_raw": auroc(scores, labels),
                      "tpr_at_1pct_fpr": all_tpr, "threshold": all_thr,
                      "threshold_source": "cal" if "all" in cal_thresh else "val",
                      "note": "all is register-imbalanced (pile 82%); report pile/coai separately (K3)"}

    # Prefer best checkpoint if it exists (select by val AUROC, not last step).
    if best_state is not None:
        torch.save(best_state, args.out / "model.pt")
        print(f"[done] saved best.pt val AUROC {best_auroc:.4f} as model.pt", flush=True)
    else:
        torch.save(model.state_dict(), args.out / "model.pt")
    tok.save_pretrained(args.out)
    (args.out / "calibration.json").write_text(json.dumps(
        {"fpr": 0.01, "per_register": {k: v["threshold"] for k, v in metrics.items()},
         "threshold_source": "cal" if cal_scores else "val"}, indent=2) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({
        "config": asdict(cfg), "lanes": lanes, "git_sha": git_sha(), "metrics": metrics,
        "val_cal_split": {"val_frac": args.val_frac, "cal_frac": args.cal_frac},
        "warmup_frac": args.warmup_frac,
        "best_val_auroc": best_auroc if best_auroc != float("-inf") else None,
        "trained_on": [str(args.spans_parquet if args.spans_parquet.exists() else args.doc_parquet)]
        if not args.smoke else ["synthetic-smoke"],
        "never_trained_on": ["eval/labels/laguna.jsonl", "eval/labels/local.jsonl", "data/coai_test.parquet"],
        "wall_seconds": round(time.time() - started, 1),
        "note": "doc head is pile resemblance, not an authorship or %-AI claim; all AUROC is imbalanced, use pile/coai",
    }, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"[done] bundle at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
