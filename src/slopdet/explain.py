"""Checkable why-slop and why-human. Never an authorship claim."""

from __future__ import annotations

import json
import re
from typing import Any

from slopdet.construction import construction_stats, over_explain_spans
from slopdet.ontology import Ontology, default_ontology_dir, load_ontology
from slopdet.report import render_hits
from slopdet.scorer import default_bundle_path, load_bundle, score_text
from slopdet.span import split_sentences
from slopdet.storyscope import storyscope_hits
from slopdet.tags import COPY, pack_style, say
from slopdet.weaklabel import label_text

# Density proxies that fire on almost all academic prose. Checkable, but not slop evidence.
REGISTER_IDS = frozenset(
    {
        "rhet_nominalization_density",
        "rhet_copula_avoidance",
    }
)

_WEEKDAY = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?\b"
)
_CLOCK = re.compile(r"\b(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)|3am|noon|midnight)\b", re.I)
_DAYPART = re.compile(r"\b(?:mornings?|afternoons?|evenings?|nights?)\b", re.I)
_CONTRACTION = re.compile(
    r"\b(?:don't|doesn't|didn't|I'm|I've|I'd|wasn't|aren't|isn't|won't|can't)\b",
    re.I,
)
_FIRST_PERSON = re.compile(r"\bI\b")
_PROPER_INNER = re.compile(
    r"(?<!\A)(?<![.!?]\s)(?<![\"\u201c\u201d(\[])\b[A-Z][a-z]{2,}\b"
)
_DIGIT = re.compile(r"\b\d+(?:[./:]\d+)*\b")


def _slop_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [h for h in hits if h["id"] not in REGISTER_IDS]


def _human_signals(text: str, stats: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(pid: str, start: int, end: int) -> None:
        key = (pid, start, end)
        if key in seen or end <= start:
            return
        seen.add(key)
        spec = COPY[pid]
        found.append(
            {
                "id": pid,
                "lane": spec["lane"],
                "unit": "span",
                "quote": text[start:end],
                "lean": spec["lean"],
                "say": spec["say"],
            }
        )

    for rx, pid in (
        (_WEEKDAY, "weekday"),
        (_CLOCK, "clock"),
        (_DAYPART, "daypart"),
        (_CONTRACTION, "spoken"),
        (_DIGIT, "number"),
    ):
        for match in rx.finditer(text):
            add(pid, match.start(), match.end())

    name_hits = 0
    for match in _PROPER_INNER.finditer(text):
        # Cap at one: repeated title-case terms are register (Mixed-Integer
        # Programming), not names. A single mid-sentence capital is a real cue.
        if name_hits >= 1:
            break
        add("name", match.start(), match.end())
        name_hits += 1

    if _FIRST_PERSON.search(text) and stats.get("n_words", 0) <= 80:
        match = _FIRST_PERSON.search(text)
        if match:
            add("first", match.start(), match.end())

    if stats.get("n_sentences", 0) >= 3 and float(stats.get("burstiness") or 0) >= 0.25:
        found.append(
            {
                "id": "burst",
                "lane": "construction",
                "lean": "human",
                "unit": "piece",
                "quote": "",
                "say": say("burst"),
            }
        )
    if int(stats.get("adjacent_contrast") or 0) >= 1:
        found.append(
            {
                "id": "contrast",
                "lane": "construction",
                "lean": "human",
                "unit": "piece",
                "quote": "",
                "say": say("contrast"),
            }
        )
    return found


def _construction_slop(text: str, stats: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if float(stats.get("recap_closure") or 0) >= 0.5:
        out.append(
            {
                "id": "recap",
                "lane": "construction",
                "lean": "slop",
                "unit": "paragraph",
                "quote": text[-min(80, len(text)) :],
                "say": say("recap"),
                "fix": say("recap"),
            }
        )
    if float(stats.get("over_explain") or 0) > 0:
        spans = over_explain_spans(text)
        quote = spans[0]["quote"] if spans else ""
        out.append(
            {
                "id": "gloss",
                "lane": "construction",
                "lean": "slop",
                "unit": "span",
                "start": spans[0]["start"] if spans else None,
                "end": spans[0]["end"] if spans else None,
                "quote": quote,
                "say": say("gloss"),
                "fix": say("gloss"),
            }
        )
    if (
        int(stats.get("n_sentences") or 0) >= 4
        and float(stats.get("burstiness") or 0) < 0.12
    ):
        out.append(
            {
                "id": "even",
                "lane": "construction",
                "lean": "slop",
                "unit": "piece",
                "quote": "",
                "say": say("even"),
                "fix": say("even"),
            }
        )
    return out


def _lean(why_slop: list[dict[str, Any]], why_human: list[dict[str, Any]]) -> str:
    if why_slop and why_human:
        return "mixed"
    if why_slop:
        return "slop"
    if why_human:
        return "human"
    return "unclear"


def _doc_lean(sentences: list[dict[str, Any]]) -> str:
    leans = {s["lean"] for s in sentences}
    if "slop" in leans and "human" in leans:
        return "mixed"
    if "slop" in leans:
        return "slop"
    if "human" in leans:
        return "human"
    return "unclear"


def _pile_resemblance(text: str, onto: Ontology) -> dict[str, Any] | None:
    path = default_bundle_path()
    if not path.is_file():
        return None
    try:
        bundle = load_bundle(path)
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    if bundle.get("ontology_sha256") != onto.sha256:
        return None
    scored = score_text(text, ontology=onto, bundle=bundle)
    return {
        "label": scored["label"],
        "text": scored["text"],
        "human_percentile": scored["human_percentile"],
        "trained_on": scored.get("trained_on"),
    }


def _sentence_record(sentence: str, ontology: Ontology) -> dict[str, Any]:
    hits = _slop_hits(label_text(sentence, ontology))
    stats = construction_stats(sentence)
    why_human = _human_signals(sentence, stats)
    why_slop = [pack_style(h) for h in hits]
    human_leaned = [h for h in why_slop if h.get("lean") == "human"]
    why_slop = [h for h in why_slop if h.get("lean") != "human"]
    why_human.extend(human_leaned)
    return {
        "text": sentence,
        "lean": _lean(why_slop, why_human),
        "why_slop": why_slop,
        "why_human": why_human,
    }


def explain(
    text: str, ontology_dir: Any | None = None, *, sentences: bool = True
) -> dict[str, Any]:
    onto = load_ontology(ontology_dir or default_ontology_dir())
    raw_hits = label_text(text, onto)
    slop_hits = _slop_hits(raw_hits)
    stats = construction_stats(text)
    why_slop = [pack_style(h) for h in slop_hits]
    # A pattern's lean is authoritative: human-lean hits (weasel, frames,
    # passive) vote human, not slop. Partition here so _lean() and the
    # slop/human tag columns agree with the per-span lean.
    human_leaned = [h for h in why_slop if h.get("lean") == "human"]
    why_slop = [h for h in why_slop if h.get("lean") != "human"]
    why_slop.extend(_construction_slop(text, stats))
    why_human = _human_signals(text, stats)
    why_human.extend(human_leaned)
    for hit in storyscope_hits(text):
        spec = COPY[hit["id"]]
        hit.setdefault("say", spec["say"])
        hit.setdefault("fix", spec["say"])
        (why_slop if hit["lean"] == "slop" else why_human).append(hit)
    if sentences:
        sentence_records = [_sentence_record(s, onto) for s in split_sentences(text)]
        lean = _doc_lean(sentence_records)
    else:
        sentence_records = []
        lean = _lean(why_slop, why_human)
    rendered = render_hits(why_slop, resemblance=_pile_resemblance(text, onto))
    rendered.update(
        {
            "lean": lean,
            "why_slop": why_slop,
            "why_human": why_human,
            "sentences": sentence_records,
            "construction": stats,
            "ontology_sha256": onto.sha256,
        }
    )
    return rendered
