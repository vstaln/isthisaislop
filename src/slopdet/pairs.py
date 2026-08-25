"""Same-prompt pairing — the single definition of "these rows are twins".

Two consumers share it so they can never disagree. `split_rows` keeps a family
whole when it cuts train/val/cal, so a human source and the machine text derived
from it can never straddle a split boundary; `build_pairs` emits the (human, ai)
index pairs the contrastive objective trains on.

Pure function module — no I/O, no torch. Deterministic output order.
"""

from __future__ import annotations

import random
from collections import Counter

# split_hint prefixes that carry a PER-PAIR key: the same value is stamped on the
# human source and on every machine text derived from it.
#
#   hc3:<question>       scripts/fetch_v2.py     fetch_hc3
#   wiki_intro:<title>   scripts/fetch_v2.py     fetch_wiki_intro
#   para:<prompt_id>     scripts/merge_all_gen.py  rewrite of a human story
#   premise:<prompt_id>  scripts/merge_all_gen.py  fresh story from a human premise
#   fictpair:<prompt_id> scripts/merge_fictpair.py WritingPrompts prompt
#
# Everything else in `split_hint` is a corpus tag that is constant across a whole
# source — 'beemo', 'raid:train-shard', 'm4:<domain>', 'subtaskC', 'v1:<register>'.
# Grouping on those would collapse an entire register into one family, so they are
# deliberately not pair keys.
PAIR_HINT_PREFIXES = ("hc3:", "wiki_intro:", "para:", "premise:", "fictpair:")


def pair_key(row: dict) -> str | None:
    """Return the family key for a row, or None when the row is unpaired.

    A '#' in the hint marks a per-prompt family ('storyscope_<split>#<prompt_id>')
    and counts as a pair key for the same reason the prefixes above do.
    """
    hint = str(row.get("split_hint") or "")
    if hint.startswith(PAIR_HINT_PREFIXES) or "#" in hint:
        return hint
    return None


def families(rows: list[dict]) -> list[list[int]]:
    """Group row indices into families: paired rows together, the rest singletons.

    Families come out ordered by first appearance so the result is deterministic.
    """
    by_key: dict[str, list[int]] = {}
    out: list[list[int]] = []
    for idx, row in enumerate(rows):
        key = pair_key(row)
        if key is None:
            out.append([idx])
            continue
        group = by_key.get(key)
        if group is None:
            by_key[key] = group = []
            out.append(group)
        group.append(idx)
    return out


def _stratum(row: dict) -> tuple[str, int]:
    return (str(row.get("register", "")), int(row.get("label", 0)))


def _targets(count: int, val_frac: float, cal_frac: float) -> tuple[int, int]:
    """val/cal row targets for one stratum, leaving at least one row for train."""
    if count <= 0:
        return 0, 0
    n_val = max(1, int(count * val_frac))
    n_cal = max(1, int(count * cal_frac))
    if n_val + n_cal >= count:
        n_val = max(1, count // 10) if count >= 10 else 1
        n_cal = 1 if count >= 3 else 0
        if n_val + n_cal >= count:
            n_cal = 0
    return n_val, n_cal


def split_rows(
    rows: list[dict],
    val_frac: float = 0.05,
    cal_frac: float = 0.05,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows into (train, val, cal), stratified per (register, label).

    cal is disjoint from val so the 1%-FPR thresholds are not fitted on the slice
    they are reported against. Pair families are assigned as a unit: a family is
    placed by the stratum it has the most rows in, and every row of that family
    follows, which is what keeps a rewrite out of val while its source is in train.
    """
    counts = Counter(_stratum(r) for r in rows)
    want_val: dict[tuple[str, int], int] = {}
    want_cal: dict[tuple[str, int], int] = {}
    for stratum, count in counts.items():
        want_val[stratum], want_cal[stratum] = _targets(count, val_frac, cal_frac)

    groups = families(rows)
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)

    got_val: Counter = Counter()
    got_cal: Counter = Counter()
    val_idx: list[int] = []
    cal_idx: list[int] = []
    train_idx: list[int] = []
    for gi in order:
        group = groups[gi]
        demand = Counter(_stratum(rows[i]) for i in group)
        # The stratum the family belongs to most; ties broken by name for determinism.
        home = min(demand, key=lambda s: (-demand[s], s))
        if got_val[home] < want_val.get(home, 0):
            bucket = val_idx
            counter = got_val
        elif got_cal[home] < want_cal.get(home, 0):
            bucket = cal_idx
            counter = got_cal
        else:
            bucket = counter = None
        if bucket is None:
            train_idx.extend(group)
        else:
            bucket.extend(group)
            counter.update(demand)

    train_idx.sort()
    val_idx.sort()
    cal_idx.sort()
    return ([rows[i] for i in train_idx],
            [rows[i] for i in val_idx],
            [rows[i] for i in cal_idx])


def build_pairs(rows: list[dict], max_pairs: int = 60000) -> list[tuple[int, int]]:
    """Return up to max_pairs (i, j) index pairs where rows[i] is the human member
    and rows[j] the machine member of the same family.

    Deterministic: families in first-appearance order, indices in input order
    within each family, humans iterated outer and machine texts inner.
    """
    out: list[tuple[int, int]] = []
    for group in families(rows):
        if len(group) < 2:
            continue
        humans = [i for i in group if rows[i].get("label") == 0]
        ais = [i for i in group if rows[i].get("label") == 1]
        for i in humans:
            for j in ais:
                out.append((i, j))
                if len(out) >= max_pairs:
                    return out
    return out
