"""Span weak-labeler. Overlapping hits from different ids all survive."""

from __future__ import annotations

from typing import Any

from slopdet.ontology import Ontology, Pattern


def _word_count(text: str) -> int:
    return len(text.split())


def label_text(text: str, ontology: Ontology, *, enabled_only: bool = True) -> list[dict[str, Any]]:
    n_words = _word_count(text)
    patterns: tuple[Pattern, ...] = (
        ontology.enabled_patterns() if enabled_only else ontology.patterns
    )
    hits: list[dict[str, Any]] = []
    for pattern in patterns:
        if n_words < pattern.min_len_words:
            continue
        for match in pattern.compiled.finditer(text):
            start, end = match.start(), match.end()
            if end <= start:
                continue
            hits.append(
                {
                    "id": pattern.id,
                    "start": start,
                    "end": end,
                    "unit": pattern.unit,
                    "lane": pattern.lane,
                    "quote": text[start:end],
                    "fix": pattern.fix,
                }
            )
    hits.sort(key=lambda h: (h["start"], h["end"], h["id"]))
    return hits
