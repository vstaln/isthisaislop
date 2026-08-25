"""The detector model — one definition shared by trainer, evaluator and exporter.

It used to be written out three times (`fine_tune_lfm.EncoderDetector`,
`eval_trained.DetectorBundle`, `export_onnx.DetectorBundle`) with *different*
token-head widths, so a checkpoint could load into the evaluator or the exporter
with a silently mismatched head under `strict=False`. The lane count now travels
with the weights: `from_checkpoint` reads it off the token head instead of
guessing.

The doc head is a pile-resemblance score calibrated per register, never a
percentage of AI-ness (docs/HANDOFF.md §1).
"""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("detector.py needs torch. Install the train extra.") from exc

from .lfm import load_encoder_body

TOKEN_HEAD_KEY = "token.weight"


class DetectorBundle(nn.Module):
    """LFM2 bidirectional body, mean-pooled doc head + per-token lane head.

    n_lanes is the number of *named* ontology lanes; the token head adds one more
    class for "no lane". n_lanes=0 is legitimate: the v2 corpus ships an empty
    spans column on purpose, so the token head stays unsupervised there.
    """

    def __init__(self, model_name: str, n_lanes: int, dtype=None):
        super().__init__()
        self.n_lanes = n_lanes
        self.body = load_encoder_body(model_name, dtype=dtype)
        hidden = self.body.config.hidden_size
        self.doc = nn.Linear(hidden, 2)
        self.token = nn.Linear(hidden, n_lanes + 1)

    def encode(self, input_ids, attention_mask):
        """Token states plus the mask-weighted mean pooling both heads read."""
        states = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return states, pooled

    def forward(self, input_ids, attention_mask):
        states, pooled = self.encode(input_ids, attention_mask)
        return self.doc(pooled), self.token(states)

    def forward_doc_embed(self, input_ids, attention_mask):
        """Pooled embedding + doc logits, for the same-prompt contrastive loss."""
        _, pooled = self.encode(input_ids, attention_mask)
        return pooled, self.doc(pooled)


def lanes_in_state_dict(state: dict) -> int:
    """Recover n_lanes from a checkpoint's token head, so callers never guess."""
    weight = state.get(TOKEN_HEAD_KEY)
    if weight is None:
        raise KeyError(f"checkpoint has no {TOKEN_HEAD_KEY}: not a DetectorBundle")
    return int(weight.shape[0]) - 1


def unwrap_state_dict(obj) -> dict:
    """Accept a bare state dict or a training checkpoint that nests one."""
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


def from_checkpoint(path: str | Path, model_name: str, device=None) -> DetectorBundle:
    """Build the bundle with the lane count the checkpoint was trained with, and
    refuse to load if any tensor is missing rather than scoring with a random head."""
    state = unwrap_state_dict(torch.load(str(path), map_location="cpu"))
    model = DetectorBundle(model_name, lanes_in_state_dict(state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} tensors missing from the checkpoint "
            f"(first: {list(missing)[:3]}). Refusing to evaluate a partly random model."
        )
    if unexpected:
        print(f"[detector] ignoring {len(unexpected)} unexpected keys "
              f"(first: {list(unexpected)[:3]})", flush=True)
    if device is not None:
        model = model.to(device)
    return model.eval()
