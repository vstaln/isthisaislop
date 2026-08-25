"""Corpus loading: schema guards, span coercion and the register allowlist."""

from __future__ import annotations

import pytest

from slopdet import corpus
from slopdet.calibrate import register_allowed


def test_v2_contrastive_registers_are_allowed():
    # scripts/merge_all_gen.py names these '<mode>_pair'; the shipped v2 parquet
    # carries 9,992 rewrite_pair and 2,811 respond_pair rows, and training used to
    # abort on them before the allowlist knew the suffix.
    for register in ("rewrite_pair", "respond_pair", "direct_pair"):
        assert register_allowed(register)


def test_every_shipped_v2_register_family_is_allowed():
    for register in ("coai", "writingprompts", "gutenberg", "blogs", "wiki_intro",
                     "wiki_intro_gpt", "hc3_finance", "hc3_finance_gpt", "beemo",
                     "beemo_ai", "raid_books", "m4_arxiv", "semeval_mixed", "fictpair"):
        assert register_allowed(register)
    assert not register_allowed("pile_v3")


def test_check_registers_rejects_unknown_names_and_labels():
    with pytest.raises(SystemExit, match="unknown register"):
        corpus.check_registers([{"register": "not_a_register", "label": 0}])
    with pytest.raises(SystemExit, match="bad label"):
        corpus.check_registers([{"register": "coai", "label": 2}])


def test_coerce_spans_tolerates_the_shapes_parquet_hands_back():
    assert corpus.coerce_spans(None) == []
    assert corpus.coerce_spans("not json") == []
    assert corpus.coerce_spans('[{"lane": "style", "start": 0, "end": 3}]') == [
        {"lane": "style", "start": 0, "end": 3}]
    assert corpus.coerce_spans([{"lane": "style"}, {"start": 1}, "junk"]) == [{"lane": "style"}]


def test_smoke_rows_are_balanced_and_paired():
    rows = corpus.smoke_rows(4)
    assert len(rows) == 8
    assert sum(r["label"] for r in rows) == 4
    assert len({r["split_hint"] for r in rows}) == 4


def _write(tmp_path, table):
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "corpus.parquet"
    pq.write_table(table, path)
    return path


def test_load_rows_reads_labels_spans_and_hints(tmp_path):
    pa = pytest.importorskip("pyarrow")
    path = _write(tmp_path, pa.table({
        "text": ["human text", "machine text"],
        "label": [0, 1],
        "register": ["blogs", "rewrite_pair"],
        "spans": [[], [{"lane": "style", "start": 0, "end": 7}]],
        "split_hint": ["", "para:1"],
    }))
    rows = corpus.load_rows(path)
    assert [r["label"] for r in rows] == [0, 1]
    assert rows[1]["spans"] == [{"lane": "style", "start": 0, "end": 7}]
    assert rows[1]["split_hint"] == "para:1"


def test_load_rows_falls_back_to_the_pile_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    path = _write(tmp_path, pa.table({"text": ["a", "b"], "pile": ["human", "ai"],
                                      "register": ["coai", "coai"]}))
    assert [r["label"] for r in corpus.load_rows(path)] == [0, 1]


def test_load_rows_refuses_a_corpus_without_registers(tmp_path):
    pa = pytest.importorskip("pyarrow")
    path = _write(tmp_path, pa.table({"text": ["a"], "label": [0]}))
    with pytest.raises(SystemExit, match="missing 'register' column"):
        corpus.load_rows(path)


def test_load_rows_refuses_a_row_with_no_label(tmp_path):
    pa = pytest.importorskip("pyarrow")
    path = _write(tmp_path, pa.table({"text": ["a"], "register": ["coai"]}))
    with pytest.raises(SystemExit, match="missing label/pile"):
        corpus.load_rows(path)


def test_resolve_prefers_a_local_file(tmp_path):
    path = tmp_path / "v2_train_labeled.parquet"
    path.write_bytes(b"")
    assert corpus.resolve(path) == path


def test_resolve_accepts_the_legacy_local_name(tmp_path):
    legacy = tmp_path / corpus.LEGACY_LOCAL_NAMES[0]
    legacy.write_bytes(b"")
    assert corpus.resolve(tmp_path / corpus.V2_TRAIN_FILE) == legacy
