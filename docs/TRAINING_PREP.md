# Training prep — T4 run

Status: data ready, script fixed, validated, notebook rebuilt for the **350M encoder**.
Train on a free Colab T4.

## What's ready

**Data** — `data/training/train_all.parquet` (713 MB, gitignored):
- 122,336 rows, 199M tokens
- columns: `text, label (0=human,1=AI), register, spans` (spans = `{lane,start,end}` char offsets, int)
- registers: coai 62,460 (balanced academic) · storyscope 36,915 (AI fiction) · gutenberg 15,104 (recent human fiction) · blogs 7,807 (human) · scp 50 (human)
- pile: 68,145 AI / 54,191 human (1.26:1)
- built by `scripts/build_training_parquet.py` (re-run after any spans change)

**Model** — `LiquidAI/LFM2.5-Encoder-350M` (newest LFM2.5 encoder, ~354M params,
bidirectional masked-LM, 15-lang claim — but measured: en + de/es/fr/it/nl/pt are
native; zh/ja/ar/hi/ru are byte-fallback, see language scan). Default in
`fine_tune_lfm.py` Config and in the notebook.

**Fixes in this session** (committed):
1. `scripts/fine_tune_lfm.py::load_rows` — crashed on empty-span rows (`array([]) or []` → numpy truth ValueError). Now coerces empty/numpy spans safely. **This would have crashed training at data load.**
2. `scripts/build_training_parquet.py` — int-coerces span offsets, skips spans with `None` offsets, adds `register` column (the trainer's per-register eval needs it; the raw spans parquets lack it).
3. **K1 (brutal review)**: `src/slopdet/labels.py::parse_label` — one shared label parser for both scripts; strict enum; raises on garbage. Previously the two scripts parsed `label`/`pile` with opposite missing-value defaults (missing→AI vs missing→human), which could silently flip thousands of labels on any data change.

**Validated end-to-end** (no GPU needed):
- `load_rows(train_all.parquet)` → 122,336 rows, 4 lanes (construction/rhetorical/storyscope/style), empty-span rows OK
- `SlopDataset` produces `input_ids/attention_mask/doc_label/token_labels` (512 tokens, lane ids populated)
- tokenizer `LiquidAI/LFM2.5-Encoder-350M` loads from HF; 350M weights load via `load_encoder_body` (smoke path)

## Training on a T4 (Colab)

1. `uv run python notebooks/build_colab_nb.py` (regenerates the notebook from repo files)
2. Open `notebooks/SlopDetector_Colab.ipynb` in Colab
3. **Runtime → Change runtime type → T4 GPU**
4. Upload `data/training/train_all.parquet` to Drive `MyDrive/isthisaislop/` (or /content)
5. **Runtime → Run all**

The notebook: mounts Drive → writes package files → locates the parquet → runs
`fine_tune_lfm.py --arch encoder --model LiquidAI/LFM2.5-Encoder-350M --spans-parquet train_all.parquet --max-len 512 --out artifacts/lfm`.

Checkpoints go to `artifacts/lfm/checkpoint.pt` (every 500 steps) — copy them out to
Drive during the run; free-T4 session death is normal.

## The training command (manual)

```bash
# upload data/training/train_all.parquet to Colab (e.g. /content/train_all.parquet),
# then from the repo root:
uv run python scripts/fine_tune_lfm.py \
  --arch encoder \
  --model LiquidAI/LFM2.5-Encoder-350M \
  --spans-parquet data/training/train_all.parquet \
  --max-len 512 \
  --out artifacts/lfm
```

- No `--doc-parquet` needed (load_rows prefers `--spans-parquet`).
- Config defaults are sane: batch 8, grad_accum 4, lr 2e-5, **1 epoch**, fp16, ckpt every 500 steps.
- **T4 warning** (handoff §3): Turing-class = no bf16, no flash-attn 2. If loss goes non-finite, the script aborts with instructions → rerun `--precision fp32` or lower `--lr`. Do NOT silence the abort.
- Outputs bundle to `--out`: `model.pt`, tokenizer, `calibration.json` (threshold per register @1% FPR), `manifest.json`.

## Before training — sanity checks (incl. brutal review findings)

The brutal review (`research/notes/brutal-review.md`) verdict is **FIX-FIRST**. K1 is fixed (shared label parser). The rest are accepted risks for a first run — do NOT let them block the T4 launch, but know them:

1. **Per-register split**: `fine_tune_lfm.main()` splits val per (register, label) at `--val-frac 0.05`. With scp having only 50 docs, its val group = 1-3 docs — AUROC on scp is a die roll. Accept as near-unevaluable or drop scp.
2. **Storyscope token dominance**: 163M of 199M tokens (82%), all AI. The model may overweight fiction. The register×label confound (K3) means "all" AUROC will be inflated by storyscope-vs-gutenberg discrimination. **Never headline the global number** — report per-register, and read coai separately.
3. **Token head lanes**: spans' `lean` is NOT used by the trainer (only lane/start/end) — the weasel/frames/passive human flips only matter for eval reports, not training. Confirm that's intended.
4. **max-len 512 vs long docs**: storyscope docs are 6.5k words — heavily truncated at 512 tokens. The doc head sees ~350 words; the token head sees >99% "no lane" tokens (K4). First-run acceptance: fine. Future fix: sliding-window chunking + token-loss reweight.
5. **fp16 on T4**: the NaN preflight abort is the guard; have `--precision fp32` ready.
6. **eval floor to beat**: CPU logistic scorer = AUC 0.8643 / acc 0.7675 on coai (in `artifacts/MANIFEST.json`). The reviewer's K2: this floor is coai-only, and the handoff's own §5 says "a single coai number is not a result" — so beat it **per register**, and add a coai-only eval of the neural model on the same 11,022 test docs for apples-to-apples.
7. **Checkpoint to Drive**: session death is normal on free Colab; `--ckpt-every 500` writes `artifacts/lfm/checkpoint.pt` — mount Drive and copy it out periodically.
8. **Calibration circularity (M2)**: thresholds come from the same 5% val slice used for metrics. First-run acceptance: fine; future: hold out a distinct calibration set.
9. **Language**: English-only by design. The 350M's zh/ja/ar/hi/ru are byte-fallback — do NOT add Chinese as a fifth register. If zh is ever a requirement, it's a separate model with a zh-native encoder.

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
