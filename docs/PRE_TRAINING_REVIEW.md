# ITAIS — Pre-Training Review Request (for an expert reviewer)

> **HISTORICAL — v1, 2026-08-18.** This is the review request that produced
> `research/notes/brutal-review.md` and the K1–K5 / M1–M4 fixes. The v1 run it
> describes happened and failed; see the V1 POSTMORTEM in `docs/HANDOFF.md`.
> The current plan is v2 — do not build from this file.

**You are being asked to critique a plan before a ~3.6h T4 training run.**
Be brutally honest. Everything below is verified against the repo (git `5eb90ff`).
The goal: catch anything that will produce a garbage model, wasted GPU time, or false confidence —
**before** we spend the run.

---

## What the product is

A browser-extension-scale AI-slop detector:
- Input: text (paste/select)
- Output: "matches AI pile vs human" verdict + **checkable why-spans** (which sentences, which pattern names)
- Constraint: must run on-device (CPU/phone), no "% AI" claims, no authorship claims (report.py guard)

## What we're doing

Fine-tune `LiquidAI/LFM2.5-Encoder-350M` (bidirectional masked-LM encoder, ~354M params, 15-lang claim
but measured: en+western-EU native, zh/ja/ar byte-fallback) into a 2-head model:
- **doc head**: mean-pooled → 2-class (AI vs human)
- **token head**: per-token lane classification (4 lanes: construction/rhetorical/storyscope/style)

Config: 1 epoch, lr 2e-5 (constant, no schedule), batch 8 × grad_accum 4 = 32, max_len 512, fp16
with fp32 fallback, checkpoint every 500 steps, AdamW wd 0.01. ~21.5k optimizer steps.

## The data (the thing we want you to scrutinize)

**`train_all.parquet` — 686,712 docs, ~527M tokens, 432,521 AI / 254,191 human (1.70:1)**

| register | docs | source | label provenance |
|---|---|---|---|
| `pile` (artem9k) | 564,376 | ai-text-detection-pile (real internet: news/blogs/essays) | `source` col: 364k AI / 200k human (shortest-first) |
| `coai` | 62,460 | arxiv abstracts vs LLM paraphrases | HF labels, balanced 31,230/31,230 |
| `storyscope` | 36,915 | 5-LLM-generated fiction | AI by construction |
| `gutenberg` | 15,104 | recent (PG≥50000) public-domain fiction | human by construction |
| `blogs` | 7,807 | scraped blogs | human by construction |
| `scp` | 50 | misc | human by construction |

Each row: `text, label (0/1), register, spans` where spans = list of
`{lane, start, end}` char offsets (pattern id/lean/quote dropped in build).

**Span labels** come from a deterministic regex ontology (`src/slopdet/explain.py`,
~40 patterns across 4 lanes). LLM lanes (mimo-v2.5, deepseek-v4-flash) were used as
*validation only* — on 130 docs they agree **42.3%**. We did NOT use LLM labels for training.

## Known issues (from a prior brutal review — verify each, and tell us what else)

### KILLER (we've fixed K1; K2-K5 open)

- **K1 — FIXED**: label parsing was inconsistent across scripts (missing→AI vs missing→human).
  Now one shared `parse_label()` (`src/slopdet/labels.py`) that raises on garbage. ✓
- **K2 — OPEN**: the "floor" (AUC 0.8643, coai test 11,022 docs, CPU logistic scorer) is
  coai-only. The neural model's eval is per-register on a 5% val slice. Are these comparable?
  What's the right way to claim "beat the floor"?
- **K3 — OPEN**: register ≡ label by construction for storyscope/gutenberg/blogs/scp/pile.
  A model could score 1.0 on "modern prose" vs 0.0 on "19th-century prose" without learning slop.
  We added pile (real internet text, both labels) + writingprompts (100k human fiction) to
  break this — is it enough? Is the "all" AUROC even meaningful, or should we only report
  per-register + coai?
- **K4 — OPEN (partially fixed)**: max_len 512 truncates storyscope docs (~6.5k words, ~9.5x the
  window). We added class-balanced token loss (rare lanes get up to 50x weight) so the token
  head learns instead of collapsing to "no lane" (>99% of tokens are class 0). But the doc head
  still sees only the first 512 tokens. Is that acceptable for a first run? Should we chunk?
- **K5 — OPEN**: QAD/GGUF quantized checkpoints rejected (wrong runtime: llama.cpp vs our
  torchscript/ONNX; quantization error worst at 1%-FPR tail). We plan INT8 dynamic ONNX export
  AFTER training with a drift check. Is that the right quantization approach for on-device?

### MAJOR

- **M1**: 1 epoch, constant 2e-5, no warmup/schedule, no checkpoint selection, no early stop.
  For 686k docs / 21.5k steps is this reasonable? What schedule would you recommend?
- **M2**: calibration thresholds (per-register @1% FPR) computed on the SAME 5% val slice used
  for reported metrics. Circular. Fix = hold out a separate calibration set? We have coai test
  (11,022 docs) as a held-out — use that?
- **M3**: English-only pipeline, undocumented. Chinese data exists on disk but is NOT in training
  (separate model needed — 350M has 0 CJK vocab). OK to ship English-only?
- **M4**: small registers (scp=50 docs → 3 val rows; blogs=7,807 → ~40 val rows) give AUROC die
  rolls. Drop them from per-register reporting? Require min N?

### Minor (acknowledged)

- m1: report.py fallback copy "Resembles the AI pile more than N% of human reference texts" is
  technically not-authorship but reads as "% AI" to users.
- m2: span `lean`/`quote` dropped in training (only lane/start/end kept) — token head must
  rediscover patterns from raw tokens.
- m3: missing register defaults to "coai" (m3) — combined with K1's old defaulting, a malformed
  parquet silently becomes coai/AI data. parse_label now raises on garbage, but register default
  is still silent.
- m4: single seed (0), no CI/bootstraps on AUROC.

## Training-time facts to sanity-check

- 350M on T4, 512 max_len, batch 8 × 4 = 32, fp16 → est **3.6h/epoch** (from 230M @ 1.1h/epoch
  for 199M tokens, scaled 1.5x for 350M). Plausible?
- Free Colab T4: ~12GB RAM, session death normal → checkpoint every 500 steps to Drive.
- fp16 NaN abort → auto-fp32 retry (Turing has no bf16, no flash-attn 2).
- The 350M tokenizer: 65,536 vocab, 8k context (we use 512), loads via
  `AutoModelForMaskedLM` → `.lfm2` body (a trap: `AutoModel` gives a random-init body; we assert
  no missing keys).

## What we want from you

1. **Ranked verdict on train-vs-fix**: is this good enough to spend 3.6h on, or is there a
   must-fix before training? Be specific.
2. **The eval question**: what is the honest, defensible way to report this model's quality
   given K2/K3 (register confound + coai-only floor)? What numbers would you trust?
3. **Hyperparameter opinion**: 1 epoch @ 21.5k steps, lr 2e-5 constant — good, or should we do
   warmup/decay/lr sweep? What's the highest-leverage change?
4. **The token head**: with class-balanced loss, is lane/start/end-only (no lean/quote) enough
   for the "why" to work? Is 4 lanes the right granularity?
5. **Anything we're blind to** — data leakage (e.g., coai test docs in train?), eval
   methodology, deployment concerns, what a smarter person would catch in 5 minutes.

**Repo**: github.com/vstaln/isthisaislop (everything public). Key files:
- `docs/HANDOFF.md` (rev 2 — the authoritative plan)
- `scripts/fine_tune_lfm.py` (trainer)
- `scripts/build_training_parquet.py` (data assembly)
- `scripts/label_artem9k.py`, `scripts/label_writingprompts.py` (span labeling)
- `src/slopdet/explain.py`, `src/slopdet/tags.py` (the ontology)
- `research/notes/brutal-review.md` (prior review — disagree with it if wrong)
- `scripts/eval_trained.py` (post-training eval + INT8 quantize probe)
