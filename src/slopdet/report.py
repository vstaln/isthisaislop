"""Hit renderer. Two lanes never merge. Never emit a percentage-of-AI string."""

from __future__ import annotations

from typing import Any

FORBIDDEN_SUBSTRINGS = (
    "% AI",
    "% ai",
    "percent AI",
    "AI-generated",
    "ai-generated",
    "written by ChatGPT",
    "written by GPT",
)


def render_hits(
    hits: list[dict[str, Any]],
    *,
    resemblance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    style = [h for h in hits if h.get("lane") != "resemblance"]
    out: dict[str, Any] = {
        "status": "ok",
        "hits": [
            {
                "id": h["id"],
                "lane": h.get("lane", "style"),
                "unit": h.get("unit", "span"),
                "quote": h.get("quote", ""),
                "say": h.get("say") or h.get("fix", ""),
                "fix": h.get("fix") or h.get("say", ""),
            }
            for h in style
        ],
        "style_summary": "Nothing matched." if not style else None,
        "resemblance": None,
    }
    if not style:
        out["style_summary"] = "Nothing matched."
    if resemblance is not None:
        pct = resemblance.get("human_percentile")
        if pct is None:
            out["resemblance"] = {
                "label": "matches_ai_pile",
                "text": resemblance.get("text", "Resemblance unavailable."),
            }
        else:
            out["resemblance"] = {
                "label": "matches_ai_pile",
                "text": (
                    f"Resembles the AI pile more than {pct:.0f}% of human reference texts."
                ),
            }
    # The guard applies to OUR copy (say/fix/summaries/resemblance/labels), never
    # to verbatim user quotes: a span that literally quotes "AI-generated" from
    # the user's own text is not a claim we make, and crashing the labeling
    # pipeline on it would be a false positive (StoryScope fiction trips this).
    # Scanned from the generated fields directly (not `str(dict)`, whose repr
    # escaping breaks substring matching on multi-line quotes).
    bits: list[str] = [str(out.get("status", "")), str(out.get("style_summary") or "")]
    for hit in out["hits"]:
        for key in ("id", "lane", "unit", "say", "fix"):
            bits.append(str(hit.get(key, "")))
    res = out.get("resemblance") or {}
    bits += [str(res.get("label", "")), str(res.get("text", ""))]
    blob = "\n".join(bits)
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in blob and "AI pile" not in bad:
            holder = next(
                (k for k, v in out.items() if bad in str(v)),
                "hits[].say/fix",
            )
            raise ValueError(f"forbidden copy leaked: {bad!r} (in {holder})")
    return out
