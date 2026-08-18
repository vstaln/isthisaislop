"""Stitch human and AI sentences so a token classifier can mark which spans."""

from __future__ import annotations

import random
import re
from typing import Any

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def stitch_docs(
    docs: list[dict[str, Any]],
    rng: random.Random,
    n_human: int = 3,
    n_ai: int = 3,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Splice human + AI sentences into mixed docs with per-sentence labels (0=human, 1=ai)."""
    human = [d["text"] for d in docs if int(d["label"]) == 0]
    ai = [d["text"] for d in docs if int(d["label"]) == 1]
    rng.shuffle(human)
    rng.shuffle(ai)
    mixed: list[dict[str, Any]] = []
    n_pairs = min(len(human), len(ai))
    if limit is not None:
        n_pairs = min(n_pairs, limit)
    for i in range(n_pairs):
        h_sents = split_sentences(human[i])
        a_sents = split_sentences(ai[i])
        if len(h_sents) < n_human or len(a_sents) < n_ai:
            continue
        sentences = [(s, 0) for s in h_sents[:n_human]] + [(s, 1) for s in a_sents[:n_ai]]
        rng.shuffle(sentences)
        text, offset, spans = "", 0, []
        for sent, lab in sentences:
            spans.append((offset, offset + len(sent), lab))
            text += sent + " "
            offset += len(sent) + 1
        mixed.append(
            {
                "id": f"stitch-{i}",
                "text": text.strip(),
                "sentences": spans,
                "source": "stitched",
            }
        )
    return mixed


def pure_docs(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Whole-document spans labeled with the source pile. Used for calibration, not training."""
    human: list[dict[str, Any]] = []
    ai: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        text = str(doc["text"])
        lab = int(doc["label"])
        rec = {
            "id": doc.get("id", f"pure-{lab}-{i}"),
            "text": text,
            "sentences": [(0, len(text), lab)],
            "source": "pure",
            "label": lab,
        }
        (human if lab == 0 else ai).append(rec)
    return human, ai


def token_labels(doc: dict[str, Any], tokenizer: Any, max_len: int) -> dict[str, list]:
    """Map per-sentence source labels to per-token labels. -100 = special/padding (ignored)."""
    enc = tokenizer(doc["text"], truncation=True, max_length=max_len, return_offsets_mapping=True)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    labels = [-100] * len(input_ids)
    offsets = enc["offset_mapping"]
    for i, (start, end) in enumerate(offsets):
        if i == 0 or i == len(offsets) - 1 or start == end:
            continue
        for s, e, lab in doc["sentences"]:
            if s <= start < e:
                labels[i] = lab
                break
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
