"""Pair families: the split must never separate a human source from its rewrite."""

from __future__ import annotations

from collections import Counter

from slopdet.pairs import build_pairs, families, pair_key, split_rows


def test_every_documented_pair_prefix_is_a_key():
    # These are the five keys the v2 build stamps; scripts/merge_all_gen.py's
    # docstring is the contract and slopdet.pairs is the only reader of it.
    for hint in ("hc3:why is the sky blue", "wiki_intro:Ada Lovelace",
                 "para:wp:1234", "premise:gut:99", "fictpair:abc"):
        assert pair_key({"split_hint": hint}) == hint


def test_corpus_tags_are_not_pair_keys():
    # Constant-per-source hints. Treating them as pair keys would drop a whole
    # register into one family and hand it wholesale to train or to val.
    for hint in ("beemo", "raid:train-shard", "m4:reddit", "subtaskC", "v1:coai", ""):
        assert pair_key({"split_hint": hint}) is None
    assert pair_key({}) is None


def test_prompt_family_hints_group_by_hash():
    assert pair_key({"split_hint": "storyscope_train#42"}) == "storyscope_train#42"


def _corpus() -> list[dict]:
    """Two paired sources whose members sit in different registers and labels,
    plus unpaired filler — the shape that made the old split leak."""
    rows = []
    for i in range(40):
        rows.append({"text": f"human hc3 {i}", "label": 0, "register": "hc3_finance",
                     "split_hint": f"hc3:q{i}"})
        rows.append({"text": f"gpt hc3 {i}", "label": 1, "register": "hc3_finance_gpt",
                     "split_hint": f"hc3:q{i}"})
        rows.append({"text": f"human story {i}", "label": 0, "register": "writingprompts",
                     "split_hint": f"para:p{i}"})
        rows.append({"text": f"rewrite {i}", "label": 1, "register": "rewrite_pair",
                     "split_hint": f"para:p{i}"})
    for i in range(60):
        rows.append({"text": f"lone human {i}", "label": 0, "register": "blogs", "split_hint": ""})
        rows.append({"text": f"lone ai {i}", "label": 1, "register": "m4_reddit",
                     "split_hint": "m4:reddit"})
    return rows


def test_families_group_pairs_and_keep_singletons_apart():
    groups = families(_corpus())
    sizes = Counter(len(g) for g in groups)
    assert sizes[2] == 80        # 40 hc3 + 40 para families
    assert sizes[1] == 120       # unpaired rows stay their own family


def test_split_never_straddles_a_pair_family():
    rows = _corpus()
    train, val, cal = split_rows(rows, 0.2, 0.2, seed=0)
    seen: dict[str, set[str]] = {}
    for name, part in (("train", train), ("val", val), ("cal", cal)):
        for row in part:
            key = pair_key(row)
            if key:
                seen.setdefault(key, set()).add(name)
    assert seen, "fixture should contain pair families"
    straddling = {k: v for k, v in seen.items() if len(v) > 1}
    assert straddling == {}


def test_split_is_a_partition():
    rows = _corpus()
    train, val, cal = split_rows(rows, 0.1, 0.1, seed=0)
    assert len(train) + len(val) + len(cal) == len(rows)
    texts = [r["text"] for r in train + val + cal]
    assert sorted(texts) == sorted(r["text"] for r in rows)


def test_split_is_deterministic_and_seed_sensitive():
    rows = _corpus()
    first = split_rows(rows, 0.1, 0.1, seed=0)
    assert [len(p) for p in first] == [len(p) for p in split_rows(rows, 0.1, 0.1, seed=0)]
    assert [r["text"] for r in first[1]] == [r["text"] for r in split_rows(rows, 0.1, 0.1, seed=0)[1]]
    assert [r["text"] for r in first[1]] != [r["text"] for r in split_rows(rows, 0.1, 0.1, seed=7)[1]]


def test_split_still_stratifies_every_register():
    rows = _corpus()
    _, val, cal = split_rows(rows, 0.2, 0.2, seed=0)
    total = Counter(r["register"] for r in rows)
    for part in (val, cal):
        got = Counter(r["register"] for r in part)
        for register, n in total.items():
            assert got[register] > 0, f"{register} missing from a split"
            assert 0.1 <= got[register] / n <= 0.35, f"{register} at {got[register]}/{n}"


def test_tiny_group_keeps_a_training_row():
    rows = [{"text": "a", "label": 0, "register": "scp", "split_hint": ""},
            {"text": "b", "label": 1, "register": "scp", "split_hint": ""}]
    train, val, cal = split_rows(rows, 0.5, 0.5, seed=0)
    assert len(train) + len(val) + len(cal) == 2
    assert cal == []


def test_build_pairs_covers_para_and_premise():
    rows = [
        {"text": "human", "label": 0, "register": "gutenberg", "split_hint": "premise:g1"},
        {"text": "machine", "label": 1, "register": "respond_pair", "split_hint": "premise:g1"},
        {"text": "src", "label": 0, "register": "writingprompts", "split_hint": "para:w1"},
        {"text": "rewrite", "label": 1, "register": "rewrite_pair", "split_hint": "para:w1"},
    ]
    assert build_pairs(rows) == [(0, 1), (2, 3)]


def test_build_pairs_ignores_unpaired_rows_and_respects_the_cap():
    rows = [{"text": "h", "label": 0, "register": "blogs", "split_hint": "beemo"},
            {"text": "a", "label": 1, "register": "beemo_ai", "split_hint": "beemo"}]
    assert build_pairs(rows) == []

    many = []
    for i in range(10):
        many.append({"text": f"h{i}", "label": 0, "register": "hc3_finance",
                     "split_hint": f"hc3:{i}"})
        many.append({"text": f"a{i}", "label": 1, "register": "hc3_finance_gpt",
                     "split_hint": f"hc3:{i}"})
    assert len(build_pairs(many, max_pairs=4)) == 4
