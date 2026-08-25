"""The lane count travels with the weights, so no caller has to guess it.

Guessing is what the three copies of this model used to do: the exporter assumed
8 lanes, the post-training evaluator recomputed them from the parquet, and
eval_proper hardcoded 4 — then loaded the checkpoint with strict=False and scored
with a partly random head.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slopdet.detector import lanes_in_state_dict, unwrap_state_dict  # noqa: E402


def _state(n_lanes: int) -> dict:
    return {"doc.weight": torch.zeros(2, 8), "token.weight": torch.zeros(n_lanes + 1, 8)}


def test_lane_count_comes_off_the_token_head():
    assert lanes_in_state_dict(_state(0)) == 0
    assert lanes_in_state_dict(_state(4)) == 4


def test_a_checkpoint_without_a_token_head_is_rejected():
    with pytest.raises(KeyError, match="token.weight"):
        lanes_in_state_dict({"doc.weight": torch.zeros(2, 8)})


def test_training_checkpoints_and_bare_state_dicts_both_load():
    bare = _state(2)
    assert unwrap_state_dict(bare) is bare
    assert unwrap_state_dict({"step": 500, "model": bare, "auroc": 0.9}) is bare
