# Training prep — T4 run

Status: data ready, script fixed, validated. Train on a free Colab T4.

## What's ready

**Data** — `data/training/train_all.parquet` (713 MB, gitignored):
- 122,336 rows, 199M tokens
- columns: `text, label (0=human,1=AI), register, spans` (spans = `{lane,start,end}` char offsets, int)
- registers: coai 62,460 (balanced academic) · storyscope 36,915 (AI fiction) · gutenberg 15,104 (recent human fiction) · blogs 7,807 (human) · scp 50 (human)
- pile: 68,145 AI / 54,191 human (1.26:1)
- built by `scripts/build_training_parquet.py` (re-run after any spans change)

**Fixes in this session** (committed):
1. `scripts/fine_tune_lfm.py::load_rows` — crashed on empty-span rows (`array([]) or []` → numpy truth ValueError). Now coerces empty/numpy spans safely. **This would have crashed training at data load.**
2. `scripts/build_training_parquet.py` — int-coerces span offsets, skips spans with `None` offsets, adds `register` column (the trainer's per-register eval needs it; the raw spans parquets lack it).

**Validated end-to-end** (no GPU needed):
- `load_rows(train_all.parquet)` → 122,336 rows, 4 lanes (construction/rhetorical/storyscope/style), empty-span rows OK
- `SlopDataset` produces `input_ids/attention_mask/doc_label/token_labels` (512 tokens, lane ids populated)
- tokenizer `LiquidAI/LFM2.5-Encoder-230M` loads from HF

## The training command

```bash
# upload data/training/train_all.parquet to Colab (e.g. /content/train_all.parquet),
# then from the repo root:
uv run python scripts/fine_tune_lfm.py \
  --arch encoder \
  --model LiquidAI/LFM2.5-Encoder-230M \
  --spans-parquet data/training/train_all.parquet \
  --max-len 512 \
  --out artifacts/lfm
```

- No `--doc-parquet` needed (load_rows prefers `--spans-parquet`).
- Config defaults are sane: batch 8, grad_accum 4, lr 2e-5, **1 epoch**, fp16, ckpt every 500 steps.
- **T4 warning** (handoff §3): Turing-class = no bf16, no flash-attn 2. If loss goes non-finite, the script aborts with instructions → rerun `--precision fp32` or lower `--lr`. Do NOT silence the abort.
- Outputs bundle to `--out`: `model.pt`, tokenizer, `calibration.json` (threshold per register @1% FPR), `manifest.json`.

## Before training — sanity checks for the smarter agent

1. **Per-register split**: `fine_tune_lfm.main()` splits val per (register, label) at `--val-frac 0.05`. With scp having only 50 docs, its val group = 1 doc — AUROC on scp will be noise. Consider dropping scp from training or accept it as near-unevaluable.
2. **Storyscope token dominance**: 163M of 199M tokens (82%). The model may overweight fiction. Consider capping storyscope docs (e.g. `--limit` in build script) or per-register sampling.
3. **Token head lanes**: spans' `lean` is NOT used by the trainer (only lane/start/end) — the weasel/frames/passive human flips only matter for eval reports, not training. Confirm that's intended.
4. **max-len 512 vs long docs**: storyscope docs are 6.5k words — heavily truncated at 512 tokens. The span head only sees the first 512 tokens' spans. Fine for the doc head (mean-pooled) but span recall on long docs is capped.
5. **fp16 on T4**: the NaN preflight abort is the guard; have `--precision fp32` ready.
6. **eval floor to beat**: CPU logistic scorer = AUC 0.8643 / acc 0.7675 on coai (in `artifacts/MANIFEST.json`). Per handoff §4, the neural model must beat the floor **per register** (fiction + blogs), not just globally.
7. **Checkpoint to Drive**: session death is normal on free Colab; `--ckpt-every 500` writes `artifacts/lfm/checkpoint.pt` — mount Drive and copy it out periodically.

## The eval data (for after training)

- `eval/labels/laguna.jsonl` (mimo, 130 docs), `eval/labels/deepseek_eval.jsonl` (deepseek, 130), `local.jsonl`
- `scripts/export_training.py` → `data/training/labeled.parquet` (380 rows, LLM + local leans)
- Per-register eval gates are in the handoff §5 (AUROC + TPR@1%FPR per register, threshold calibrated per register).

## Rebuild data (if the smarter agent changes ontology/tags)

```bash
# re-derive spans for all corpora with current ontology (chunked, bounded RAM):
uv run python scripts/rebuild_spans.py data/training/spans_coai_train.parquet data/training/spans_storyscope_train.parquet data/training/spans_gutenberg_train.parquet data/training/spans_blogs_train.parquet data/training/spans_scp_train.parquet --workers 4 --chunk 400 --write
# then rebuild the combined training parquet:
uv run python scripts/build_training_parquet.py
```

⚠️ `rebuild_spans.py` is **slow** on storyscope (6.5k-word docs, ~1.5 docs/s at 4 workers → hours). Use `--workers 4` (6+ thrashes swap on 9.7GB). Only rebuild storyscope if the ontology change affects it.
