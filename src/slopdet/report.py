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
                "fix": h.get("fix", ""),
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
    blob = str(out)
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in blob and "AI pile" not in bad:
            raise ValueError(f"forbidden copy leaked: {bad!r}")
    return out
