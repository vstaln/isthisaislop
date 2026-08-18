"""StoryScope narrative checks (arXiv:2604.03136, COLM 2026).

Deterministic proxies for the paper's core human-vs-AI narrative features.
Every hit carries a verbatim quote from the text; nothing is LLM judgment.
AI-elevated features land as why-slop, human-elevated as why-human.
"""

from __future__ import annotations

import re
from typing import Any

PAPER = "storyscope-2604.03136"

_MORALIZE = re.compile(
    r"\b(?:the (?:real|true|actual) (?:lesson|meaning|point)|"
    r"what (?:she|he|they) (?:really )?(?:learned|understood|realized) (?:was|is)|"
    r"the (?:lesson|takeaway) (?:was|is)|"
    r"it (?:was|had always been) about|"
    r"(?:she|he|they) learned that|"
    r"in the end,? (?:she|he|they) (?:knew|understood|learned))\b",
    re.I,
)
_SENSORY = re.compile(
    r"\b(?:the scent of|the smell of|the taste of|the feel of|the warmth of|"
    r"the chill of|the air (?:smelled|tasted|felt)|"
    r"the sound of (?:rain|water|wind))\b",
    re.I,
)
_CAUSAL = re.compile(
    r"\b(?:as a result,?|which (?:led|led to|meant that)|"
    r"this (?:caused|prompted|triggered)|consequently|and so,? (?:she|he|they))\b",
    re.I,
)
_REALIZE = re.compile(
    r"\b(?:(?:she|he|they) (?:finally )?(?:realized|understood) (?:that|the|what)|"
    r"(?:in|at) (?:the end|last),? (?:she|he|they) (?:knew|understood|realized)|"
    r"it was then that (?:she|he|they) (?:understood|realized|knew))\b",
    re.I,
)
_INTRO = re.compile(
    r"\b[A-Z][a-z]{2,} (?:was|is|had) (?:a|an|the) [a-z]+ "
    r"(?:man|woman|boy|girl|teenager|engineer|detective|teacher|soldier|doctor)"
    r"(?: with| who)?\b"
)
_AGENCY = re.compile(
    r"\b(?:she|he|they) (?:decided|chose|resolved|made the decision) (?:to|that)\b",
    re.I,
)
_READER = re.compile(
    r"\bthe reader\b|\bYou (?:were|stood|walked|watched|heard|felt)\b|\bPicture this\b",
    re.I,
)
_QUOTE = re.compile(r'"([^"]{2,})"|\u201c([^\u201d]{2,})\u201d')


def _hit(pid: str, start: int, end: int, text: str, lean: str) -> dict[str, Any]:
    return {
        "id": pid,
        "lane": "storyscope",
        "unit": "span",
        "start": start,
        "end": end,
        "quote": text[start:end],
        "lean": lean,
        "paper": PAPER,
    }


def _rx_hits(
    text: str, rx: re.Pattern[str], pid: str, lean: str
) -> list[dict[str, Any]]:
    return [_hit(pid, m.start(), m.end(), text, lean) for m in rx.finditer(text)]


def _dialogue_hit(text: str) -> list[dict[str, Any]]:
    matches = [m for m in _QUOTE.finditer(text)]
    if len(matches) < 2:
        return []
    spoken = sum(len(m.group(1) or m.group(2)) for m in matches)
    if len(text) > 0 and spoken / len(text) >= 0.35:
        first = matches[0]
        return [_hit("dialogue", first.start(), first.end(), text, "human")]
    return []


def storyscope_hits(text: str) -> list[dict[str, Any]]:
    """Deterministic narrative features. Slop-elevated first, then human-elevated."""
    hits: list[dict[str, Any]] = []
    for rx, pid in (
        (_MORALIZE, "moralize"),
        (_SENSORY, "sensory"),
        (_CAUSAL, "causal"),
        (_REALIZE, "realize"),
        (_INTRO, "intro"),
        (_AGENCY, "agency"),
    ):
        hits.extend(_rx_hits(text, rx, pid, "slop"))
    hits.extend(_rx_hits(text, _READER, "reader", "human"))
    hits.extend(_dialogue_hit(text))
    hits.sort(key=lambda h: (h["start"], h["end"], h["id"]))
    return hits
