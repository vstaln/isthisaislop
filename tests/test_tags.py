"""Model emits a tag + quote. Copy is hardcoded."""

from slopdet.explain import REGISTER_IDS
from slopdet.ontology import load_ontology
from slopdet.tags import COPY, hydrate, hydrate_lanes, pack_style, say, style_word

def test_catalog_says_are_one_liners():
    for pid, spec in COPY.items():
        n = len(spec["say"].split())
        assert n <= 14, (pid, spec["say"], n)
        assert "\n" not in spec["say"]


def test_hydrate_drops_llm_paragraph():
    hit = hydrate(
        {
            "id": "thematic_explicitness",
            "quote": "In conclusion, we hope this helps.",
            "reason": "The narrator explicitly states the lesson in a long paragraph of analysis.",
            "lean": "human",
        }
    )
    assert hit is not None
    assert hit["id"] == "moral"
    assert hit["say"] == "States the lesson. Let the fact stand."
    assert hit["lean"] == "slop"
    assert "reason" not in hit
    assert "paragraph" not in hit["say"]


def test_pack_style_uses_ontology_fix_not_llm_prose():
    hit = pack_style(
        {
            "id": "throat_clearing",
            "quote": "Here's the thing",
            "fix": "Cut the opener. Start on the point.",
            "lane": "style",
            "start": 0,
            "end": 16,
        }
    )
    assert hit["id"] == "opener"
    assert hit["pattern"] == "throat_clearing"
    assert hit["say"] == "Cut the opener. Start on the point."
    assert hit["id"].isalpha()


def test_every_enabled_pattern_maps_to_a_catalog_word():
    onto = load_ontology()
    missing = []
    for pattern in onto.enabled_patterns():
        if pattern.id in REGISTER_IDS:
            continue
        word = style_word(pattern.id)
        if word not in COPY:
            missing.append((pattern.id, word))
    assert missing == []


def test_unknown_id_is_dropped():
    assert hydrate({"id": "vibes", "quote": "something"}) is None


def test_hydrate_lanes_splits_by_catalog_not_model_lean():
    out = hydrate_lanes(
        {
            "lean": "mixed",
            "style": [{"id": "glue_bans", "quote": "leverage", "reason": "essay"}],
            "construction": [{"id": "named_anchors", "quote": "Thursday", "reason": "essay"}],
        }
    )
    assert out["style"] == [
        {
            "id": "glue",
            "quote": "leverage",
            "lean": "slop",
            "lane": "style",
            "say": say("glue"),
        }
    ]
    assert out["construction"][0]["id"] == "anchor"
    assert "reason" not in out["construction"][0]
