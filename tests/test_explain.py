"""Explainer: named why-slop + why-human + per-sentence lean. No authorship claims."""

from slopdet.explain import explain
from slopdet.report import FORBIDDEN_SUBSTRINGS


SLOP = (
    "Here's the thing, in today's competitive landscape we leverage robust "
    "pipelines. In conclusion, experts agree it's a pivotal moment."
)
HUMAN = (
    "Thursday mornings at the clinic were empty. Half the early slots sat unused, "
    "and the monthly average hid it."
)
MIXED = (
    "Here's the thing, we leverage robust tools. Thursday mornings at the clinic were empty."
)


def test_slop_sample_names_checkable_hits():
    out = explain(SLOP)
    ids = {h["id"] for h in out["why_slop"]}
    assert "opener" in ids
    assert "glue" in ids
    for hit in out["why_slop"]:
        assert hit["quote"]
        assert hit["fix"]
        assert hit["quote"] in SLOP
    assert out["lean"] == "slop"


def test_human_sample_names_checkable_cues():
    out = explain(HUMAN)
    assert out["why_slop"] == []
    assert out["style_summary"] == "Nothing matched."
    quotes = " ".join(h["quote"] for h in out["why_human"])
    says = " ".join(h["say"] for h in out["why_human"])
    blob = quotes + " " + says
    assert "Thursday" in blob or "clinic" in blob
    assert out["lean"] == "human"


def test_mixed_text_labels_sentences_separately():
    out = explain(MIXED)
    assert out["lean"] == "mixed"
    assert len(out["sentences"]) == 2
    assert out["sentences"][0]["lean"] == "slop"
    assert out["sentences"][1]["lean"] == "human"
    assert any(h["id"] == "opener" for h in out["sentences"][0]["why_slop"])
    assert out["sentences"][1]["why_slop"] == []


def test_never_authorship_or_percent_ai():
    for text in (SLOP, HUMAN, MIXED):
        blob = str(explain(text))
        for bad in FORBIDDEN_SUBSTRINGS:
            if "AI pile" in bad:
                continue
            assert bad not in blob, bad
        assert "written by ChatGPT" not in blob
        assert "This looks human." not in blob
        assert "AI-generated" not in blob


def test_academic_density_proxies_are_not_why_slop():
    text = (
        "This paper introduces and studies a declarative framework for updating "
        "views over indefinite databases. An indefinite database is a database "
        "with null values that are represented by a single null constant."
    )
    out = explain(text)
    ids = {h["id"] for h in out["why_slop"]}
    assert "rhet_nominalization_density" not in ids
    assert "rhet_copula_avoidance" not in ids
