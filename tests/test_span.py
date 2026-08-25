"""Span stitching: mixed human/AI sentences with per-sentence source labels."""

from __future__ import annotations

import random

import pytest

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


def test_span_offsets_slice_back_to_the_quote(tmp_path):
    """Span start/end must slice the source text back to the quote verbatim.

    This used to read `/tmp/label_artem9k/part_*.parquet` — one machine's scratch
    output from a v1 labeling run — and return silently when it was absent, so it
    asserted nothing anywhere else. It builds its own fixture now.
    """
    import sys
    from pathlib import Path

    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_training_parquet import load_spans

    text = "We leverage robust pipelines. Thursday at 3pm I counted 41 chairs."
    source = tmp_path / "spans.parquet"
    pd.DataFrame([{
        "text": text,
        "pile": 1,
        "spans": [
            {"lane": "style", "start": text.index("leverage"),
             "end": text.index("leverage") + len("leverage"), "quote": "leverage"},
            {"lane": "construction", "start": text.index("41 chairs"),
             "end": text.index("41 chairs") + len("41 chairs"), "quote": "41 chairs"},
            # dropped: no lane, and offsets missing
            {"lane": None, "start": 0, "end": 2, "quote": "We"},
            {"lane": "style", "start": None, "end": None, "quote": ""},
        ],
    }]).to_parquet(source)

    rows = load_spans(source, "pile").to_dict("records")
    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["register"] == "pile"

    spans = rows[0]["spans"]
    assert len(spans) == 2, "spans without a lane or offsets must be dropped"
    for span, quote in zip(spans, ("leverage", "41 chairs")):
        assert 0 <= span["start"] < span["end"] <= len(text)
        assert text[span["start"]:span["end"]] == quote
