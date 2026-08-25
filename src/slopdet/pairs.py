"""Same-prompt contrastive pairing: group rows by pair key, emit (human, ai) index pairs.

Pure function module — no I/O, no torch. Deterministic output order.
"""

from __future__ import annotations


def _pair_key(row: dict) -> str:
    """Derive the pair key for one row.

    Rows with an explicit split_hint ('hc3:<q>', 'wiki_intro:<title>',
    'fictpair:<prompt_id>') pair on the hint itself. Everything else
    (e.g. raid) falls back to (register, first 60 normalized chars of text);
    such keys only yield pairs when both labels exist under them, which the
    grouping below enforces implicitly.
    """
    hint = row.get("split_hint") or ""
    if hint.startswith(("hc3:", "wiki_intro:", "fictpair:")):
        return hint
    text = row.get("text") or ""
    norm = " ".join(str(text).lower().split())[:60]
    return f"derived:{row.get('register', '')}:{norm}"


def build_pairs(rows: list[dict], max_pairs: int = 60000) -> list[tuple[int, int]]:
    """Return up to max_pairs (i, j) index pairs where rows[i].label == 0,
    rows[j].label == 1, and both rows share the same derived pair key.

    Deterministic: keys sorted ascending, indices in input order within each
    key; humans iterated outer, AIs inner. Caps at max_pairs total.
    """
    humans: dict[str, list[int]] = {}
    ais: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        label = row.get("label")
        key = _pair_key(row)
        if label == 0:
            humans.setdefault(key, []).append(idx)
        elif label == 1:
            ais.setdefault(key, []).append(idx)

    out: list[tuple[int, int]] = []
    for key in sorted(set(humans) & set(ais)):
        for i in humans[key]:
            for j in ais[key]:
                out.append((i, j))
                if len(out) >= max_pairs:
                    return out
    return out
