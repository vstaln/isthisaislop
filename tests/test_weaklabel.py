"""20 hand-written sentences. Assert span recovery for planted patterns."""

from __future__ import annotations

from slopdet.ontology import load_ontology
from slopdet.weaklabel import label_text


CASES = [
    ("Here's the thing, the eval is the product.", "throat_clearing"),
    ("It's worth noting that the loss dropped.", "phrase_worth_noting"),
    ("In today's competitive market, speed wins.", "phrase_in_todays"),
    ("At its core, the model copies residuals.", "phrase_at_its_core"),
    ("When it comes to detection, spans matter.", "phrase_when_it_comes_to"),
    ("In conclusion, we restated the intro.", "recap_ending"),
    ("Let's dive in to the training loop.", "phrase_lets_dive_in"),
    ("Experts agree the method works.", "weasel_attribution"),
    ("The launch, highlighting the team's commitment, shipped.", "superficial_analysis"),
    ("The app serves as a centralized hub for files.", "fake_strong_verb"),
    ("What if I told you the cache fits on Drive?", "rhetorical_setup"),
    ("Certainly, I can help with that.", "opener_certainly"),
    ("Additionally, the student is 40M parameters.", "opener_additionally"),
    ("As an AI, I cannot browse your disk.", "opener_as_an_ai"),
    ("Please don't hesitate to reach out anytime.", "phrase_please_dont_hesitate"),
    ("I hope this email finds you well.", "phrase_i_hope_this_finds_you"),
    ("This is a testament to careful labeling.", "a_testament_to"),
    ("We should leverage the RAID train split.", "ban_delve_class"),
    ("The platform is robust and seamless.", "ban_puffery_adj"),
    ("In other words, the two lanes never merge.", "interpretive_metadiscourse"),
]


def test_planted_spans_recover_expected_id() -> None:
    onto = load_ontology()
    missed = []
    for text, expected in CASES:
        ids = {h["id"] for h in label_text(text, onto)}
        if expected not in ids:
            missed.append((expected, text, sorted(ids)))
    assert not missed, f"missed {len(missed)} planted ids: {missed[:5]}"


def test_hit_offsets_slice_back_to_quote() -> None:
    onto = load_ontology()
    text = "Here's the thing, RAID-test is never touched."
    hits = [h for h in label_text(text, onto) if h["id"] == "throat_clearing"]
    assert hits
    h = hits[0]
    assert text[h["start"] : h["end"]] == h["quote"]
