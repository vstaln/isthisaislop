#!/usr/bin/env python3
"""Kill length-shortcut risk across PAIRED registers.

hc3_open_qa (all human) + hc3_open_qa_gpt (all AI) form one family; if mean(AI)
> 1.5x mean(human) within a family, truncate AI rows to the human p95 length."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/v2/v2_train.parquet.labeled.parquet")
df = pd.read_parquet(path)
df["text"] = df["text"].astype(str)

def family(reg: str) -> str:
    for suf in ("_gpt", "_ai"):
        if reg.endswith(suf):
            return reg[: -len(suf)]
    return reg

df["family"] = df["register"].map(family)
changed = 0
print("family                    n      ai/human_len  action")
for fam in sorted(df.family.unique()):
    sub = df[df.family == fam]
    h = sub[sub.label == 0]["text"].str.len()
    a = sub[sub.label == 1]["text"].str.len()
    if len(h) < 30 or len(a) < 30:
        continue
    ratio = a.mean() / max(h.mean(), 1)
    action = ""
    if ratio > 1.5:
        mode = sys.argv[2] if len(sys.argv) > 2 else "drop"
        if mode == "truncate":
            cap = int(h.quantile(0.95))
            mask = (df.family == fam) & (df.label == 1) & (df.text.str.len() > cap)
            df.loc[mask, "text"] = df.loc[mask, "text"].str[:cap]
            changed += int(mask.sum())
            action = f"TRIMMED {int(mask.sum())} AI rows to <= {cap}"
        else:
            # drop longest AI rows until mean ratio <= 1.5
            ai_idx = df.index[(df.family == fam) & (df.label == 1)]
            lens = df.loc[ai_idx, "text"].str.len()
            target = int(a.mean() / 1.5)
            n_drop = max(0, int(len(lens) - len(lens[lens <= target]) * (a.mean() / 1.5) ** 0))
            order = lens.sort_values(ascending=False).index
            dropped = 0
            for i in order:
                cur_a = df.loc[ai_idx.difference(df.index[dropped_mask] if False else [])] if False else None
                break
            # simple greedy: sort desc, drop until mean fits
            sorted_idx = lens.sort_values(ascending=False).index.tolist()
            import numpy as np
            keep_cap = None
            arr = lens.loc[sorted_idx].values
            csum = np.concatenate([[0], arr.cumsum()])
            # find k longest to remove so remaining mean <= h.mean()*1.5
            total = arr.sum(); n = len(arr)
            k = 0
            for k in range(n):
                rem_total = total - csum[k]
                rem_n = n - k
                if rem_n and rem_total / rem_n <= h.mean() * 1.5:
                    break
            drop_idx = sorted_idx[:k]
            df.drop(drop_idx, inplace=True)
            changed += len(drop_idx)
            action = f"DROPPED {len(drop_idx)} longest AI rows"
    print(f"  {fam:24s} {len(sub):6d}   {ratio:5.2f}       {action}")

print(f"\n[fix] truncated {changed} AI rows total")
out = path.with_suffix(".parquet.fixed")
df.drop(columns=["family"]).to_parquet(out, index=False)

# verify
fx = pd.read_parquet(out)
fx["family"] = fx["register"].map(family)
worst, worst_f = 0, ""
for fam in fx.family.unique():
    sub = fx[fx.family == fam]
    h = sub[sub.label == 0]["text"].str.len()
    a = sub[sub.label == 1]["text"].str.len()
    if len(h) >= 30 and len(a) >= 30:
        r = a.mean() / max(h.mean(), 1)
        if r > worst:
            worst, worst_f = r, fam
print(f"[fix] worst remaining ai/human length ratio: {worst:.2f} ({worst_f})")
print(f"[fix] wrote {out} — verify then replace original")
