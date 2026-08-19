"""Shared label parsing for training data.

K1 fix: `scripts/fine_tune_lfm.py` and `scripts/build_training_parquet.py` used to
parse the same `label`/`pile` columns with opposite missing-value defaults
(missing -> AI in one, missing -> human in the other). A malformed or renamed
column could silently flip thousands of labels between the two entry points.

Both scripts now call `parse_label` so a doc has exactly one interpretation
everywhere: explicit `label` wins; else `pile` with a strict enum; else the
caller's explicit default. Anything unrecognized raises.
"""

from __future__ import annotations

# pile values -> label (0 = human, 1 = AI). Accepts ints, floats, bools, strings.
_PILE_TO_LABEL: dict[str, int] = {"0": 0, "1": 1, "0.0": 0, "1.0": 1,
                                  "human": 0, "ai": 1, "false": 0, "true": 1}


def parse_label(rec: dict, default: int) -> int:
    """Return the 0/1 label for a training row, or raise on an unrecognized value.

    Precedence: explicit ``label`` column > ``pile`` column > ``default``.
    """
    if "label" in rec and rec["label"] is not None:
        try:
            return int(rec["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unrecognized label {rec['label']!r}") from exc
    if "pile" in rec and rec["pile"] is not None:
        try:
            return _PILE_TO_LABEL[str(rec["pile"]).strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unrecognized pile value {rec['pile']!r}") from exc
    return int(default)
