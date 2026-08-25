#!/usr/bin/env python3
"""Smoke tests for the v2 pipeline — plain asserts, no pytest, no network.

Covers: spans.validate_row accept/reject, Deduper behavior, doc_summary
math, and pure helpers in fetch_v2 (norm/dedup keys, clean_rows,
balance_lengths, cap_register, stratified_holdout).

RUNTIME-TESTED: everything in this file (pure functions only).
NOT runtime-tested here: all network/dataset paths in fetch_v2.py,
label_v2_batch.py multiprocessing + ontology join, label_v2_llm.py
OpenRouter calls. Those are marked with comments in their files.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from slopdet import spans  # noqa: E402
import fetch_v2 as fv  # noqa: E402


def good_span(start: int = 0, end: int = 5, lean: str = "slop") -> dict:
    return {"id": "test_rule", "lane": "style", "lean": lean,
            "start": start, "end": end, "quote": "hello",
            "why": "Rule: cut it", "fix": "cut it"}


def test_validate_span() -> None:
    spans.validate_span(good_span())  # accept
    d = good_span()
    del d["why"]
    for bad, why in [
        ({}, "missing all keys"),
        (good_span() | {"start": -1}, "negative start"),
        (good_span() | {"end": 3, "start": 5}, "end <= start"),
        (good_span() | {"lean": "other"}, "bad lean"),
        (good_span() | {"quote": 5}, "wrong quote type"),
        ("not a dict", "not a dict"),
    ]:
        try:
            spans.validate_span(bad)
        except ValueError:
            continue
        raise AssertionError(f"validate_span accepted: {why}")
    print("ok validate_span")


def test_doc_summary_math() -> None:
    text = "abcdefghij" * 10  # 100 chars
    s1 = good_span(0, 20)             # 20 slop chars -> 0.20 slop-heavy
    s2 = good_span(30, 40, lean="human")
    s3 = good_span(50, 55)            # 5 more slop
    summ = spans.doc_summary(text, [s1, s2, s3])
    assert summ["slop_density"] == 0.25, summ          # 25/100
    assert summ["human_density"] == 0.10, summ         # 10/100
    assert summ["verdict"] == "slop-heavy", summ
    assert summ["lanes_fired"] == {"style": 2}, summ   # only slop spans counted
    # thresholds
    t = spans.doc_summary(text, [good_span(0, 6)])     # 0.06
    assert t["verdict"] == "slop-tinged", t
    t = spans.doc_summary(text, [good_span(0, 5)])     # 0.05 -> not > 0.05
    assert t["verdict"] == "clean", t
    assert spans.doc_summary(text, [])["verdict"] == "clean"
    print("ok doc_summary math")


def test_validate_row() -> None:
    row = {"text": "hello world", "label": 1, "register": "r", "spans": [],
           "source_dataset": "x", "generator": "g", "generation_method": "direct",
           "decoding": "", "split_hint": ""}
    spans.validate_row(row)  # accept
    spans.validate_row(row | {"spans": [good_span(0, 5)]})  # offsets in range
    for bad, why in [
        (row | {"label": 2}, "label not binary"),
        (row | {"label": True}, "label bool"),
        (row | {"text": 5}, "text not str"),
        (row | {"register": ""}, "empty register"),
        (row | {"spans": "no"}, "spans not list"),
        (row | {"spans": [good_span(0, 99)]}, "offset past end"),
        (row | {"generation_method": "vibes"}, "bad generation_method"),
    ]:
        try:
            spans.validate_row(bad)
        except ValueError:
            continue
        raise AssertionError(f"validate_row accepted: {why}")
    # quote mismatch
    try:
        spans.validate_row(row | {"spans": [good_span(0, 5) | {"quote": "bye"}]})
    except ValueError:
        pass
    else:
        raise AssertionError("validate_row accepted quote mismatch")
    print("ok validate_row")


def test_deduper() -> None:
    d = fv.Deduper()
    assert d.add("Hello   world foo") is True
    assert d.add("hello world foo") is False          # exact-normalized dup
    long_a = "hello world foo" + "x" * 200 + "A" * 100
    long_b = "hello world foo" + "x" * 200 + "B" * 100  # same 120-char prefix, different tail
    assert d.add(long_a) is True
    assert d.add(long_b) is False                       # 120-prefix dup
    assert d.add("completely different text here") is True
    print("ok Deduper")


def test_clean_rows() -> None:
    rows = [
        fv._mkrow("short", 0, "r", source="s", generator="human", method="human"),
        fv._mkrow("x" * 300, 1, "r", source="s", generator="g", method="direct"),
        fv._mkrow("x" * 300, 1, "r", source="s", generator="g", method="direct"),  # dup
    ]
    out = fv.clean_rows(rows, fv.Deduper())
    assert len(out) == 1 and out[0]["label"] == 1, out
    # head-truncate
    long_row = fv._mkrow("y" * 10_000, 0, "r", source="s", generator="human", method="human")
    out = fv.clean_rows([long_row])
    assert len(out[0]["text"]) == fv.MAX_CHARS
    print("ok clean_rows")


def test_balance_and_cap() -> None:
    # register 'r': human lengths 1000-1100; AI row at 5000 must be dropped
    rows = []
    for i in range(20):
        rows.append(fv._mkrow("a" * (1000 + i * 5), 0, "r", source="s",
                              generator="human", method="human"))
    rows.append(fv._mkrow("b" * 5000, 1, "r", source="s", generator="g", method="direct"))
    rows.append(fv._mkrow("b" * 1050, 1, "r", source="s", generator="g", method="direct"))
    out, dropped = fv.balance_lengths(rows)
    assert dropped == {"r": 1}, dropped
    assert all(len(r["text"]) < 4000 for r in out)
    # register with no human side passes through
    only_ai = [fv._mkrow("c" * 9000, 1, "solo", source="s", generator="g", method="direct")]
    out, dropped = fv.balance_lengths(only_ai)
    assert len(out) == 1 and not dropped
    # cap: big register trimmed to <=25% of total, small register survives
    rows = ([fv._mkrow("a" * 300, 0, "big", source="s", generator="human", method="human")]
            * 6000) + [fv._mkrow("b" * 300, 0, "small", source="s", generator="human",
                                 method="human")] * 10
    out, trimmed = fv.cap_register(rows)
    big = sum(1 for r in out if r["register"] == "big")
    small = sum(1 for r in out if r["register"] == "small")
    assert small == 10, small                       # small register never killed
    assert big == max(int(len(rows) * 0.25), 1000), big  # 25% cap, floor 1000
    print("ok balance/cap")


def test_stratified_holdout() -> None:
    rows = ([fv._mkrow("a" * 300, 0, "r0", source="s", generator="human", method="human")]
            + [fv._mkrow("b" * 300, 1, "r1", source="s", generator="g", method="direct")] * 99)
    hold, rest = fv.stratified_holdout(rows, 0.03)
    assert len(hold) >= 2, len(hold)   # >=1 per stratum
    assert len(hold) + len(rest) == len(rows)
    print("ok stratified_holdout")


if __name__ == "__main__":
    test_validate_span()
    test_doc_summary_math()
    test_validate_row()
    test_deduper()
    test_clean_rows()
    test_balance_and_cap()
    test_stratified_holdout()
    print("ALL SMOKE TESTS PASSED")
