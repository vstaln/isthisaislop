from slopdet.calibrate import human_percentile, threshold_at_fpr
from slopdet.construction import construction_stats
from slopdet.report import render_hits
from slopdet.verify import unverified_payload


def test_empty_hits_say_nothing_matched() -> None:
    out = render_hits([])
    assert out["style_summary"] == "Nothing matched."
    assert out["hits"] == []
    assert "% AI" not in str(out)
    assert "AI-generated" not in str(out)


def test_resemblance_names_the_comparison_class() -> None:
    out = render_hits([], resemblance={"human_percentile": 94.0})
    assert "AI pile" in out["resemblance"]["text"]
    assert "human reference" in out["resemblance"]["text"]
    assert "87% AI" not in str(out)


def test_construction_burstiness_on_varied_sentences() -> None:
    text = "Hi. This second sentence is quite a bit longer than the first one, on purpose."
    stats = construction_stats(text)
    assert stats["n_sentences"] >= 2
    assert stats["burstiness"] > 0


def test_verify_fail_closed_payload() -> None:
    payload = unverified_payload("mismatch:student.safetensors")
    assert payload["status"] == "unverified_artifact"
    assert payload["hits"] == []
    assert payload["resemblance"] is None


def test_one_percent_fpr_threshold() -> None:
    human = [0.1] * 99 + [0.9]
    t = threshold_at_fpr(human, 0.01)
    assert t >= 0.1
    assert human_percentile(0.95, human) >= 99.0
