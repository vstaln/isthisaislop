"""Shared span schema + row validation for the v2 training pipeline.

A span is a dict with exactly the keys in SPAN_KEYS. `why`/`fix` are joined
from the frozen ontology by scripts/label_v2_batch.py; fetchers emit empty
span lists. Every training row must survive validate_row before it lands in
a parquet (enforced by label_v2_batch and checked by test_v2_smoke).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

SPAN_KEYS = {"id", "lane", "lean", "start", "end", "quote", "why", "fix"}

# runtime-tested (see scripts/test_v2_smoke.py)
_SPAN_TYPES: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "lane": str,
    "lean": str,
    "start": int,
    "end": int,
    "quote": str,
    "why": str,
    "fix": str,
}

_LEANS = ("slop", "human")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_span(d: Any) -> None:
    """Raise ValueError unless d is a well-formed span dict."""
    # runtime-tested
    if not isinstance(d, dict):
        raise ValueError(f"span must be a dict, got {type(d).__name__}")
    missing = SPAN_KEYS - set(d.keys())
    if missing:
        raise ValueError(f"span missing keys: {sorted(missing)}")
    for k, want in _SPAN_TYPES.items():
        v = d[k]
        ok = _is_int(v) if want is int else isinstance(v, want)
        if not ok:
            raise ValueError(f"span[{k!r}] must be {want.__name__}, got {v!r}")
    if d["lean"] not in _LEANS:
        raise ValueError(f"span lean must be one of {_LEANS}, got {d['lean']!r}")
    if not (_is_int(d["start"]) and d["start"] >= 0):
        raise ValueError(f"span start must be >= 0, got {d['start']!r}")
    if not (_is_int(d["end"]) and d["end"] > d["start"]):
        raise ValueError(f"span end must be > start, got {d['end']!r}")


def doc_summary(text: str, spans: list[dict]) -> dict:
    """Coverage summary: densities are chars-covered / len(text).

    verdict: slop-heavy if slop_density > 0.15, slop-tinged if > 0.05,
    else clean. Runtime-tested (math asserted in scripts/test_v2_smoke.py).
    """
    n = max(len(text), 1)
    slop_chars = 0
    human_chars = 0
    lanes_fired: Counter[str] = Counter()
    for s in spans:
        if not isinstance(s, dict):
            continue
        cov = max(0, min(s.get("end", 0), len(text))) - max(0, s.get("start", 0))
        if s.get("lean") == "slop":
            slop_chars += cov
            lanes_fired[s.get("lane", "?")] += 1
        elif s.get("lean") == "human":
            human_chars += cov
    slop_density = slop_chars / n
    human_density = human_chars / n
    if slop_density > 0.15:
        verdict = "slop-heavy"
    elif slop_density > 0.05:
        verdict = "slop-tinged"
    else:
        verdict = "clean"
    return {
        "slop_density": round(slop_density, 4),
        "human_density": round(human_density, 4),
        "lanes_fired": dict(lanes_fired),
        "verdict": verdict,
    }


def validate_row(row: dict) -> None:
    """Validate one full training-row dict, offsets included.

    Requires text(str)/label(0|1)/register(str)/spans(list); validates
    metadata columns when present. Raises ValueError on any violation.
    Runtime-tested (accept + reject cases in scripts/test_v2_smoke.py).
    """
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError(f"row text must be str, got {type(text).__name__}")
    if row.get("label") not in (0, 1) or isinstance(row.get("label"), bool):
        raise ValueError(f"row label must be 0 or 1, got {row.get('label')!r}")
    if not isinstance(row.get("register"), str) or not row["register"]:
        raise ValueError(f"row register must be non-empty str, got {row.get('register')!r}")
    spans = row.get("spans")
    if not isinstance(spans, list):
        raise ValueError(f"row spans must be list, got {type(spans).__name__}")
    for i, s in enumerate(spans):
        try:
            validate_span(s)
        except ValueError as e:
            raise ValueError(f"span #{i}: {e}") from e
        if s["end"] > len(text) or s["start"] >= len(text):
            raise ValueError(
                f"span #{i} offsets [{s['start']}:{s['end']}] out of range for text len {len(text)}"
            )
        if text[s["start"]:s["end"]] != s["quote"]:
            raise ValueError(
                f"span #{i} quote does not match text slice "
                f"[{s['start']}:{s['end']}]"
            )
    gen_method = row.get("generation_method")
    if gen_method is not None and gen_method not in ("direct", "paraphrase", "rewrite", "human"):
        raise ValueError(f"generation_method must be direct|paraphrase|rewrite|human, got {gen_method!r}")
