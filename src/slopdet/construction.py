"""Cheap deterministic construction stats. No spaCy, no GPU."""

from __future__ import annotations

import math
import re
from typing import Any

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARA_SPLIT = re.compile(r"\n\s*\n")
_WORD = re.compile(r"[A-Za-z']+")
_PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")
_DIGIT = re.compile(r"\d")
_DATE = re.compile(
    r"\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}|\d{4})\b",
    re.I,
)
_SUBORD = re.compile(
    r"\b(?:because|although|though|unless|while|whereas|if|when|after|before|since)\b",
    re.I,
)
_CLOSURE = re.compile(
    r"\b(?:in conclusion|ultimately|overall|to sum up|in summary|to conclude)\b",
    re.I,
)
_OVER_EXPLAIN = re.compile(
    r"\b(?:the key point is|as you can see|this distinction matters|in other words|"
    r"highlighting|underscoring|reflecting|showcasing)\b",
    re.I,
)


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _para_features(para: str) -> list[float]:
    sents = _sentences(para)
    lengths = [len(s.split()) for s in sents] or [0]
    words = _WORD.findall(para.lower())
    n = max(len(words), 1)
    mean_len = sum(lengths) / max(len(lengths), 1)
    ttr = len(set(words)) / n
    comma = para.count(",") / n
    sub = len(_SUBORD.findall(para)) / n
    mean_word = sum(len(w) for w in words) / n
    return [mean_len, ttr, comma, sub, mean_word]


def over_explain_spans(text: str) -> list[dict[str, Any]]:
    """Verbatim spans of over-explain phrases, e.g. 'the key point is'."""
    return [
        {"start": m.start(), "end": m.end(), "quote": text[m.start() : m.end()]}
        for m in _OVER_EXPLAIN.finditer(text)
    ]


def construction_stats(text: str) -> dict[str, Any]:
    sents = _sentences(text)
    lengths = [len(s.split()) for s in sents]
    mean = (sum(lengths) / len(lengths)) if lengths else 0.0
    if len(lengths) >= 2:
        var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        std = math.sqrt(var)
        burstiness = std / mean if mean else 0.0
        adjacent_contrast = sum(
            1 for a, b in zip(lengths, lengths[1:]) if abs(a - b) >= 20
        )
    else:
        burstiness = 0.0
        adjacent_contrast = 0

    paras = _paragraphs(text)
    vecs = [_para_features(p) for p in paras]
    if len(vecs) >= 2:
        sims = [
            _cosine(vecs[i], vecs[j])
            for i in range(len(vecs))
            for j in range(i + 1, len(vecs))
        ]
        evenness = sum(sims) / len(sims)
    else:
        evenness = 0.0

    if paras:
        last = paras[-1]
        earlier_openers = " ".join(_sentences(p)[:1][0] if _sentences(p) else "" for p in paras[:-1])
        last_words = set(_WORD.findall(last.lower()))
        earlier_words = set(_WORD.findall(earlier_openers.lower()))
        overlap = len(last_words & earlier_words) / max(len(last_words), 1)
        recap_closure = overlap + (0.5 if _CLOSURE.search(last) else 0.0)
    else:
        recap_closure = 0.0

    n_words = max(len(text.split()), 1)
    over_explain = 1000.0 * len(_OVER_EXPLAIN.findall(text)) / n_words

    portable = 0
    for sent in sents:
        if not (_PROPER.search(sent) or _DIGIT.search(sent) or _DATE.search(sent)):
            portable += 1
    portability = portable / max(len(sents), 1)

    return {
        "burstiness": round(burstiness, 6),
        "adjacent_contrast": int(adjacent_contrast),
        "evenness": round(evenness, 6),
        "recap_closure": round(recap_closure, 6),
        "over_explain": round(over_explain, 6),
        "portability": round(portability, 6),
        "n_sentences": len(sents),
        "n_paragraphs": len(paras),
        "n_words": n_words,
    }
