from slopdet.ontology import load_ontology
from slopdet.scorer import CONSTRUCTION_KEYS, featurize


def test_featurize_width_matches_enabled_patterns():
    onto = load_ontology()
    ids = [p.id for p in onto.enabled_patterns()]
    vec = featurize("Thursday mornings at the clinic were empty.", onto, ids)
    assert len(vec) == len(ids) + len(CONSTRUCTION_KEYS) + 1
    assert all(isinstance(x, float) for x in vec)
