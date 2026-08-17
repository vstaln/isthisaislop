"""Linear heads on a frozen student (or on a bag-of-features vector)."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("heads.py needs torch. Install the train extra.") from exc


@dataclass
class HeadConfig:
    input_dim: int
    n_lexical: int
    n_rhetorical: int
    n_jspace: int = 512
    n_construction: int = 5
    hidden: int = 128


class Heads(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        self.lexical = nn.Linear(cfg.input_dim, cfg.n_lexical)
        self.rhetorical = nn.Linear(cfg.input_dim, cfg.n_rhetorical)
        self.construction = nn.Linear(cfg.input_dim, cfg.n_construction)
        self.jspace = nn.Linear(cfg.input_dim, cfg.n_jspace)
        self.contrast = nn.Linear(cfg.input_dim, 1)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "lexical": self.lexical(pooled),
            "rhetorical": self.rhetorical(pooled),
            "construction": self.construction(pooled),
            "jspace": self.jspace(pooled),
            "contrast": self.contrast(pooled).squeeze(-1),
        }
