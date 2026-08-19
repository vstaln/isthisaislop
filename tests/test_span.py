"""Span stitching: mixed human/AI sentences with per-sentence source labels."""

from __future__ import annotations

import random
from slopdet.span import pure_docs, stitch_docs


def _docs() -> list[dict]:
    return [
        {"text": "Human one. Human two. Human three. Human four.", "label": 0},
        {"text": "Also human. More human. Still human. Yes human.", "label": 0},
        {"text": "AI one. AI two. AI three. AI four.", "label": 1},
        {"text": "Also AI. More AI. Still AI. Yes AI.", "label": 1},
    ]


def test_stitch_mixes_human_and_ai_sentences():
    mixed = stitch_docs(_docs(), random.Random(0), n_human=2, n_ai=2)
    assert mixed
    labs = {lab for _, _, lab in mixed[0]["sentences"]}
    assert labs == {0, 1}


def test_calibration_docs_are_pure_source_not_stitched():
    human, ai = pure_docs(_docs())
    assert human and ai
    assert all(d["label"] == 0 for d in human)
    assert all(d["label"] == 1 for d in ai)
    assert all("sentences" in d for d in human + ai)
    assert all({lab for _, _, lab in d["sentences"]} == {0} for d in human)
    assert all({lab for _, _, lab in d["sentences"]} == {1} for d in ai)


def test_span_offsets_match_quotes():
    """Span start/end must slice the source text back to the quote verbatim."""
    import json

    import pandas as pd

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_training_parquet import load_spans

    parts = sorted(Path("/tmp/label_artem9k").glob("part_*.parquet"))
    if not parts:
        return  # labeling not running; skip silently
    merged = pd.concat(pd.read_parquet(p) for p in parts[:3]).reset_index(drop=True)
    merged = merged[["id", "text", "label", "pile", "spans", "slop_tags", "human_tags"]]
    test_path = Path("/tmp/test_artem9k_offsets.parquet")
    merged.to_parquet(test_path)
    df = load_spans(test_path, "pile")
    bad = 0
    checked = 0
    for rec in df.to_dict("records"):
        txt = rec["text"]
        for s in rec["spans"]:
            checked += 1
            if not (0 <= s["start"] < s["end"] <= len(txt)):
                bad += 1
    assert bad == 0, f"{bad}/{checked} spans out of bounds"
    assert checked > 0
