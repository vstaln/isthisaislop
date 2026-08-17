"""Tiny Fast-style student. Token-level hidden states for distillation."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("student.py needs torch. Install the train extra.") from exc


@dataclass
class StudentConfig:
    vocab_size: int
    pad_token_id: int
    max_length: int = 256
    token_embed_dim: int = 128
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_dim: int = 1024
    dropout: float = 0.1
    output_dim: int = 2560


class Student(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.token_embed_dim, padding_idx=cfg.pad_token_id)
        self.embed_proj = (
            nn.Linear(cfg.token_embed_dim, cfg.d_model)
            if cfg.token_embed_dim != cfg.d_model
            else nn.Identity()
        )
        self.pos_embed = nn.Embedding(cfg.max_length, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.mlp_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.embed_proj(self.token_embed(input_ids)) + self.pos_embed(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        hidden = self.norm(hidden)
        return self.out(hidden)

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token = self.forward(input_ids, attention_mask)
        mask = attention_mask.to(token.dtype).unsqueeze(-1)
        return (token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
