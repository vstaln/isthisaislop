# HANDOFF — ITAIS: train the detector

**Revision:** 2026-08-25 (rev 3) · **Authoritative.** `docs/PRE_TRAINING_REVIEW.md` and the two files
under `docs/superpowers/plans/` are historical: they specify a roberta-base doc classifier and a Colab
T4 run that this revision supersedes.
**Expires 2026-09-25** — after that, re-derive from `git log`, `artifacts/MANIFEST.json` and the
rebuild commands in §4 rather than trusting the STATUS block.

**What changed in rev 3 (code cleanup pass, no training run):** the train/val/cal split is now
pair-safe; the AUROC used by the post-training evaluator was broken and is replaced; the v2 corpus
now loads itself from HF; `eval_proper.py` is folded into `eval_trained.py`; the Colab path is gone.
Details in §9. **The v1 post-mortem numbers in this file were produced by the broken AUROC and must
not be trusted** — see §9.1.

**Fact classes used below:** MEASURED (a command in this repo produced it — command named) ·
CITED (external source + date checked) · ESTIMATE (arithmetic, not run) · CONJECTURED (belief, with
the experiment that would settle it) · STALE-BY (expires on a named condition).

---

## STATUS

| | |
|---|---|
| Shipping today | deterministic lanes (415 ontology ids + construction + storyscope), CPU logistic `matches_ai_pile`, span stitch, calibration record |
| Neural model | none trained for v2. Trainer exists and is smoke-tested: `scripts/fine_tune_lfm.py` |
| Export path | **gate passed** (§3.3): LFM2 encoder → ONNX → INT8, 230.4 MB, MEASURED |
| Data | v2 is **public** — `vstalingrady/itais/v2_train_labeled.parquet`, 222,067 rows, and the trainer downloads it when `data/` is empty (MEASURED: loads in 3.4 s from a fresh clone, 27 registers, guards pass) |
| Baseline number to beat | coai AUC 0.8643 / acc 0.7675, CPU logistic (`artifacts/MANIFEST.json`, MEASURED 2026-08-19) |
| Blockers | (B1) no GPU on the cleanup machine — v2 has never been trained; (B2) target phone/chipset for the on-device latency budget |

## NEXT ACTIONS (in order; each ends in a committed artifact)

1. **Memory probe** on the rented card before committing to a full run:
   `uv run python scripts/fine_tune_lfm.py --arch encoder --max-steps 500 --max-len 2048 --batch-size 4 --precision bf16`
   It prints peak CUDA allocation at the end. Size `--batch-size`/`--grad-accum` from that number.
2. **Train v2** (§3.1–3.2): `uv run python scripts/fine_tune_lfm.py --arch encoder --out artifacts/lfm_v2`
   plus whatever the probe says. The corpus resolves itself; no upload step.
   (Validate edits first with `--smoke`, which needs no corpus: MEASURED, 3 steps in ~2.5 min on 4
   cloud vCPU with the 350M body.)
3. **Evaluate per register, not globally** (§5), in one step:
   `uv run python scripts/eval_trained.py --ckpt artifacts/lfm_v2/model.pt --quantize`
   Read `cross_register.json` before the headline — see §9.1 for why.
4. **Re-measure the v1 post-mortem** if v1 weights still exist, since its numbers came from a broken
   AUROC (§9.1). If they do not, delete the numbers rather than keep quoting them.

## STOP CONDITIONS

- No AUC without the register slice it was measured on and the FPR the threshold was set at.
- No claim that the neural model is better until it beats the §NEXT-1 floor on the **fiction and blog**
  slices, not only on coai.
- Copy rules are load-bearing product behaviour, not style: never "% AI", never "AI-generated", never
  "written by ChatGPT", never an authorship claim. Enforced by `slopdet.report.FORBIDDEN_SUBSTRINGS`
  + tests (MEASURED: `uv run python -m pytest -q` → 67 passed, 2026-08-25).
- If ONNX/on-device export of the chosen backbone fails (§3.3), the backbone is wrong regardless of
  how good its AUC is. Do not train around a dead deployment path.

---

## 1. What the product claims

Select text → verdict + why, in two independent lanes that never merge into one number:

1. **Checkable hits** — verbatim quoted span + pattern id + hardcoded fix. Falsifiable by eye.
2. **Resemblance** — calibrated `matches_ai_pile` against a *named* human reference class. Not
   authorship, not a percentage of AI-ness.

`lean` ∈ {slop, human, mixed, unclear} is an evidence balance over sentences, not a classifier verdict.

## 2. Invariants (do not break)

- Every hit carries a verbatim quote. Empty or fabricated quotes are bugs (`gloss` and `name` were
  fixed for exactly this).
- Human cues (number/name/burst) are **non-discriminative on academic prose** — they must not drive
  `lean` on coai-register text.
- Thresholds are per-register (§5), never one global cut.

## 3. Model decision

### 3.1 Recommendation: the LFM2 **encoder**, not the 1.2B decoder

Rev 1 of this doc locked `LiquidAI/LFM2.5-1.2B` (base, not Instruct) with
`AutoModelForSequenceClassification` + LoRA over the last-token hidden state. Same family, wrong
member. Liquid ships bidirectional encoders on the LFM2 backbone, built for exactly this job
(CITED, model card, checked 2026-08-18):

| | LFM2.5-Encoder-230M | LFM2.5-Encoder-350M | LFM2.5-1.2B (rev 1 choice) |
|---|---|---|---|
| Type | bidirectional MLM encoder | bidirectional MLM encoder | causal decoder |
| Params | ~229.7M | ~354.5M | ~1.17B |
| Context | 8,192 | 8,192 | 32,768 |
| Vocab / hidden | 65,536 / 1024 | 65,536 / 1024 | 65,536 |
| 17-task fine-tune mean | 79.29 ±1.02 | 81.02 ±1.00 | not benchmarked as a classifier |
| INT8 on device | ~230 MB (ESTIMATE) | ~355 MB (ESTIMATE) | ~1.2 GB (ESTIMATE) |

For reference on that 17-task GLUE/SuperGLUE/multilingual mean: ModernBERT-large (395M) scores 81.68,
mDeBERTa-v3 (280M) 80.37, ModernBERT-base (149M) 78.19. The 350M encoder is ModernBERT-large-class
quality at 350M params, and Liquid's card claims it matches or beats ModernBERT throughput with a
long-context edge on CPU.

Three reasons the encoder wins *for this product specifically*:

- **The product needs per-token labels.** `why_slop` spans and per-sentence `lean` are token
  classification. A causal decoder sees only left context at each position; the encoder sees the whole
  selection bidirectionally in one pass. Rev 1 handled this by planning a *second* model
  (`AutoModelForTokenClassification`) — the encoder does doc + span from one body with two heads.
- **It fits the device.** ~230 MB INT8 versus ~1.2 GB. The app is meant to be free, local and
  installable; a gigabyte of weights is a different product.
- **It fits the free T4.** ESTIMATE, 6ND arithmetic over the 50M-token mix: 230M encoder at 512
  tokens ≈ 1.1 h/epoch; 1.2B LoRA at 2048 tokens ≈ 5–7 h/epoch — more than one free Colab session,
  before counting attention overhead at 2048.

**Deciding experiment (do not settle this by argument):** both paths are implemented behind
`scripts/fine_tune_lfm.py --arch {encoder,decoder}`. Train the 230M encoder first because it is the
cheap one; if it clears §5's gates, the 1.2B never needs to run.

### 3.2 Training recipe (encoder path)

- Body: **`slopdet.lfm.load_encoder_body`**, never `AutoModel`. MEASURED trap (reproduce by loading
  both auto-classes with `output_loading_info=True`; recorded in `artifacts/MANIFEST.json`):
  `AutoModel.from_pretrained("LiquidAI/LFM2.5-Encoder-230M", trust_remote_code=True)` — the snippet the
  model card gives for downstream heads — returns a body with **every tensor freshly initialized**
  (embedding std 0.0200 vs 0.1011 when loaded; all params reported MISSING, checkpoint keys `lfm2.*`
  reported UNEXPECTED), and then trains and evaluates without erroring. Loading
  `AutoModelForMaskedLM` and taking `.lfm2` loads correctly and reproduces the card's documented
  mask-fill output (`The capital of France is …` → Paris). `load_encoder_body` does that and asserts
  no body tensor was initialized from scratch, so the failure cannot come back silently.
  Heads are ours (~30 lines): pooled doc head + per-token lane head.
- Two heads on one body, trained multi-task: **doc head** (mean-pooled, 2-class, `matches_ai_pile`)
  and **token head** (per-token, ontology-lane labels from `data/training/spans_*.parquet`).
  The token head predicts *named lanes*, so its output lands in the same vocabulary as the
  deterministic lane and can be cross-checked against it.
- max_len 512 for v1 (8k is available; long selections are windowed). fp16 AMP with an explicit NaN
  preflight, because Turing has no bf16 and LFM2 is bf16-native (CITED: LFM2 cards list bf16;
  FlashAttention-2 has no Turing support, upstream points T4 users at a partial-support fork,
  checked 2026-08-18). Flash-attn is therefore off on Colab; SDPA/eager only.
- 1 epoch first, then read the val curve. Rev 1's "grokking doesn't occur in this regime" is
  CONJECTURED — treat the curve as the authority.
- LoRA is optional on a 230M encoder; full fine-tune fits (ESTIMATE: fp32 weights+grads+Adam ≈ 3.7 GB).

### 3.3 Export gate — PASSED

LFM2 interleaves gated short convolutions with grouped-query attention, which is not the well-trodden
BERT graph, so the deployment risk was retired first. MEASURED, `uv run python scripts/export_onnx.py
--probe` (real weights, untrained heads; artifact `artifacts/export_probe.json`):

| | fp32 ONNX | INT8 dynamic |
|---|---|---|
| Size | 919.2 MB | **230.4 MB** |
| Max doc-logit drift vs torch | 8.9e-8 | 0.147 |
| Max token-logit drift vs torch | 3.2e-5 | 2.75 |
| Latency @512 tok | 564 ms | 371 ms |
| Latency @1024 tok | 1108 ms | 743 ms |

Read that as: the graph exports faithfully (fp32 drift is numerical noise) and INT8 gets the artifact
to 230 MB, which is a shippable size. Two honest caveats. Latency was measured on 4 threads of a cloud
vCPU, so it is a **proxy, not a phone number** — B2 exists to replace it. And the INT8 logit drift is
large enough that it must be re-checked as an **AUROC delta on trained weights** (NEXT-4) rather than
assumed harmless; if it costs accuracy, the ladder is per-channel/static QDQ with a calibration set,
then INT8 weights with fp32 activations, then a smaller `--max-len`.
The `dynamo` ONNX exporter needs `onnxscript` installed; the script falls back to the legacy
TorchScript exporter, which is deprecated but produced the faithful graph above.
If a future backbone fails this gate, the fallback order is ExecuTorch → llama.cpp/GGUF (decoder-only,
loses the token head) → roberta-base span model, which exports trivially; its trainer lived in
`notebooks/colab_pipeline.py` and was deleted with the Colab path in rev 3 (recover it from git if
that fallback is ever needed — `slopdet.span` still has the stitching it depended on).

## 4. Data assets — and how to rebuild them

`data/` is gitignored, so this table describes state that exists on **one machine**. Every row needs a
rebuild command; that is what makes the table a handoff instead of an inventory.

| Path | Contents | Rebuild |
|---|---|---|
| `data/coai_train.parquet` | 62,460 docs (31,230 AI / 31,230 human), cols `text,label,model_name` | downloaded on demand by `scripts/train_cpu_scorer.py::load_coai` |
| `data/training/spans_coai_train.parquet` | 62,460 rows, cols `text,pile,slop_tags,human_tags,spans`; 172,169 spans, 0 non-verbatim | `scripts/label_coai_batch.py` (~3 min, 8 cores) |
| `data/training/labeled.parquet` | 250 eval docs (130 AI / 120 human), LLM lean + local spans | `scripts/export_training.py` from `eval/labels/*.jsonl` (committed) |
| `data/raw/storyscope/` | 5 AI stories per prompt (GPT-5.4, DeepSeek V3.2, Kimi K2.5, Gemini 3 Flash, Claude 4.6); human side excluded for copyright | `scripts/fetch_datasets.py --job storyscope` |
| `data/raw/gutenberg_fiction/` | ~393k public-domain chunks, human fiction | `--job gutenberg_fiction` |
| `data/raw/blogs/` | 19,320 Schler blog files — **raw HTML/XML, needs text extraction** | `--job blogs` |
| `data/raw/writingprompts/` | modern human short fiction (~600 MB) | `--job writingprompts` |
| `data/raw/scp/` | CC-BY-SA modern fiction (~116 MB) | `--job scp` |

**Fix owed here:** `artifacts/MANIFEST.json` is the one artifact path git tracks — put row counts and
sha256 per file in it, so a second machine can prove it rebuilt the same corpus.

**Register coverage is the real data problem.** coai is arxiv abstracts vs LLM paraphrases: academic
register, low slop density. The product runs on blogs, emails and fiction. StoryScope covers AI
fiction; Gutenberg/WP/SCP cover human fiction; blogs cover the register the word "slop" was coined
for and are still unextracted. Training on coai and gating on coai measures the easiest slice we own.

## 5. Evaluation — per register, at a fixed FPR

Report per slice, every run: **coai (academic paraphrase) · AI fiction (StoryScope) · human fiction
(Gutenberg / WP / SCP) · blogs · short text (<50 words) · hybrid**. For each: AUROC, TPR@1%FPR,
threshold, and the count of docs behind it.

- The operating threshold is calibrated **per register** (`slopdet.calibrate.calibration_record`).
  One global 1%-FPR cut on a coai-shaped human slice will not hold on blogs — the doc's own note that
  human cues are non-discriminative on academic prose is the same problem seen from the other side.
- Gates: beat §NEXT-1's CPU-logistic floor on fiction and blogs; AUROC ≥ 0.95 and TPR@1%FPR reported
  on coai (not the only slice); short-text slice may legitimately return `unclear` rather than a score
  — report the refusal rate instead of forcing a verdict.
- Every exported bundle carries `manifest.json` with `trained_on`, `never_trained_on`, per-slice
  numbers, threshold per register, git sha, and seed.

## 6. Environment and gotchas (things that cost real time)

- Keys: env var **names** only — see `.env.example`. Rev 1 of this file printed a key prefix and the
  path of the credential store; that was the wrong call in a committed document.
- `pkill -f <pattern>` matches the calling shell → self-kill. Use `pkill -f 'label_lagun[a]'`.
- Long jobs: `setsid nohup … < /dev/null & disown`, or shell teardown kills them.
- `explain()` is slow cold (~0.7 s); `lru_cache` on `load_ontology` and `scorer.load_bundle` fixed it.
  `sentences=False` skips the per-sentence pass.
- Blogs are raw HTML — extract before use.
- The storyscope lane is untested at scale; blast StoryScope through the labeler next.
- Turing-class cards (T4) have no bf16 and no flash-attn 2 — `--precision fp32` is the fallback the
  NaN preflight tells you to use. Rented boxes die without warning: `--ckpt-every` writes atomically
  and resume restores optimizer/scheduler/scaler/RNG, so a restart is not a reset. Set `HF_TOKEN` and
  the checkpoint also pushes to `vstalingrady/lfm-ckpt` every 1000 steps.
- LFM2 encoders need `trust_remote_code=True`, and `AutoModel` silently hands back random weights —
  always go through `slopdet.lfm.load_encoder_body` (§3.2).
- `scripts/fine_tune_lfm.py --smoke` runs the whole training path on synthetic data with no corpus
  (MEASURED: 3 steps in ~2.5 min on 4 cloud vCPU with the 350M body) — use it to validate edits
  before spending GPU hours.
- Tests: `uv run python -m pytest -q` → 67 pass (MEASURED, this revision). `pytest` is a default
  dependency group, so a fresh clone needs no `--extra` to run them.
- `uv run` does not install the `train` extra. Anything that imports torch needs
  `uv sync --extra train` first.

## 7. License posture (affects "free forever")

Code is MIT. Ontology blurbs: CC BY-SA 4.0 (`patterns.wikipedia.yaml`) and Apache-2.0 (antislop
phrases) — already isolated and noted. **The base model is not MIT:** LFM Open License v1.0 is
Apache-2.0-shaped with a commercial-use limitation that ends free commercial rights above $10M annual
revenue, and derivatives stay under it, with no copyleft on your fine-tune (CITED: Liquid AI license
page + docs, checked 2026-08-18). Fine at this scale; record it in `THIRD_PARTY_NOTICES.md` and in the
exported bundle's manifest, and keep `Qwen/Qwen3-0.6B` noted as the Apache-2.0 escape hatch. Human
corpora carry their own terms (SCP is CC-BY-SA, WritingPrompts is Reddit user content) — those affect
redistribution of *data*, not of trained weights, but the manifest should say what went in.

## 8. Why this document is shaped like this

Rules for every future revision, from reviewing rev 1:

1. Status, next actions and stop conditions in the first screen.
2. Every fact carries a class (MEASURED / CITED / ESTIMATE / CONJECTURED / STALE-BY). "Already made,
   user-approved" is a decision record, not evidence.
3. **Never point at a transcript.** Rev 1's central next step was "write `scripts/fine_tune_lfm.py`
   (the inline example given in this session's transcript is the spec)" — the spec died with the
   session. Specs live in the repo as runnable code.
4. Local-only state gets rebuild commands and hashes, or it is not handed off.
5. Secrets by reference: env var names, never a value prefix.
6. One authoritative plan. Competing plan docs get a superseded header the moment they lose.
7. Targets name their baseline. "AUC ≥ 0.95" without the current number is a wish.
8. Expiry date plus a re-derivation procedure.

---

## V1 POSTMORTEM (2026-08-21) — numbers RETRACTED 2026-08-25, conclusion still stands

The verdict is unchanged: v1 final weights are an era-and-register classifier, not a
slop detector, and the HF model card keeps its DO-NOT-USE warning. The *evidence* is
not usable. Every AUROC in `artifacts/eval_proper/report.json` came from
`eval_trained.auroc`, which summed input positions instead of score ranks and so
returned a number that does not depend on the model at all (§9.1). Read the old
figures this way:

| quoted | what it actually measured |
|---|---|
| "all cross-register pairs 1.00" | the matrix concatenated humans then AIs, so positives always sat last → the broken formula returns exactly 1.00 for any model |
| "coai strict AUROC 0.5003" | coai rows arrive interleaved by label → the same formula returns ≈0.5 for any model |
| "heldouts 0.75–0.86" | label ordering inside each jsonl file, nothing else |

What survives on its own evidence: mid-training val 0.9998 was an unstratified lie,
and the mix was 72% pile. The rest needs re-measuring with the fixed metric if the
v1 weights still exist; if they do not, delete the numbers rather than keep quoting
them.

Root causes (all addressed in the v2 spec): pile length confound (72% of mix),
storyscope AI-only anthology prompts, head-truncation at 512, no paired same-topic
data, unstratified val metric.

## V2 STATE (ready to launch)

All figures below are MEASURED on the shipped parquet, 2026-08-25, by loading
`vstalingrady/itais/v2_train_labeled.parquet` through `slopdet.corpus.load_rows`.

- **Corpus**: 222,067 rows — 115,571 human / 106,496 AI — across 27 registers,
  storyscope+pile-free, with provenance columns. Public on HF; the trainer downloads
  it when `data/v2/v2_train_labeled.parquet` is absent, so there is no upload step.
  Holdouts separated: stratified / unseen-model (cohere-chat) / paraphrase /
  mixed-authorship.
- **Pair keys**: 5 prefixes carry a per-pair key and `slopdet.pairs` is the only
  reader of them — `hc3:` (11,185 rows), `wiki_intro:` (26,672), `para:` (17,292),
  `premise:` (5,484), `fictpair:` (0 in the shipped file; generation never finished).
  `beemo`, `raid:train-shard`, `m4:<domain>`, `v1:<register>` are corpus tags, **not**
  pair keys — grouping on them would collapse a whole register into one family.
  Total joinable pairs: 25,424.
- **Spans**: 11,837 rows (5.3%) still carry regex-ontology spans in 4 lanes
  (style 17,407 · rhetorical 7,892 · storyscope 1,083 · construction 202),
  concentrated in coai (6,685 rows) and writingprompts (1,178) as carry-over from the
  v1 parquet. The design intent is that spans are unused — the ontology fires on 79%
  of human rows vs 72% of AI rows, worse than chance — but the trainer still weights
  the token head at 0.5 by default. `--token-loss-weight 0` is the setting that
  matches the intent; it is not the default only because changing it silently would
  be a training-semantics change nobody asked for.
- **Pending data**: fictpair generation (ox-alpha × 3 personas on WP prompts; ~146/15k
  stories done, resumable via scripts/generate_ai_stories.py) → merge with
  scripts/merge_fictpair.py (also stamps human hints for pairing).
- **Optimizer**: AdamW (Muon implemented per DeepSeek-V4 spec behind --optimizer muon,
  untested at scale — experiment only).
- **Test vault**: holdouts + laguna/local/deepseek are FINAL-REPORTING-ONLY, reached
  through `eval_trained.py --final-report`. Never select checkpoints against them.

## V2 LAUNCH CHECKLIST (next instance)

```bash
# 1. boot the box, clone, install
git clone https://github.com/vstaln/isthisaislop && cd isthisaislop
uv sync --extra train
export HF_TOKEN=...            # optional: HF rate limits + checkpoint backup

# 2. prove the graph before spending GPU time (no corpus needed)
uv run python -m pytest -q
uv run python scripts/fine_tune_lfm.py --arch encoder --smoke --contrastive 0.5
#    expect: [ctr] pairs=N with N > 0, then 3 steps and a bundle

# 3. memory probe — sizes the real run, prints peak CUDA allocation
uv run python scripts/fine_tune_lfm.py --arch encoder --max-steps 500 \
  --max-len 2048 --batch-size 4 --precision bf16 --out artifacts/probe

# 4. launch (user approval)
uv run python scripts/fine_tune_lfm.py --arch encoder \
  --max-len 2048 --batch-size 4 --grad-accum 9 --precision bf16 \
  --contrastive 0.5 --epochs 1 --lr 2e-5 --ckpt-every 250 --out artifacts/lfm_v2

# 5. evaluate — one step, per register, calibrated at 1% FPR on the cal slice
uv run python scripts/eval_trained.py --ckpt artifacts/lfm_v2/model.pt --quantize
```

The corpus resolves itself in steps 3–5: local `data/v2/v2_train_labeled.parquet` if
present, else the public HF copy. There is no upload step and no `--spans-parquet` to
remember.

**Success bar**: paired slices (hc3_*, wiki_intro*, rewrite_pair vs its human source)
≥ 0.90 · vault files ≥ 0.85 · unseen-generator holdout not collapsed · and the
cross-register matrix **not** uniformly ≈1.00, which is the register-shortcut
signature that sank v1.

**Data gaps carried over, unfixed** (they need a rebuild, and the data phase is
frozen): M4 came back AI-only in the original fetch, RAID landed only the abstracts
and books domains, and fictpair generation stopped at ~146/15k stories, so the
corpus has zero `fictpair:` rows. Check bucket sampling before trusting either M4 or
RAID as a register.

---

## 9. Rev 3 cleanup — what changed in the code

No training ran. Everything here is a correctness or plumbing change, verified by
`uv run python -m pytest -q` (67 passed) and
`uv run python scripts/fine_tune_lfm.py --arch encoder --smoke`.

### 9.1 The post-training AUROC was broken

`eval_trained.auroc` — which `eval_proper.py` imported and used for every number it
reported — computed:

```python
order = sorted(range(len(scores)), key=lambda i: -scores[i])
ranks = [i + 1 for i in order]                        # order[k] + 1, not the rank
sum_pos_ranks = sum(r for i, r in zip(order, ranks) if labels[i])
```

`ranks[k]` is `order[k] + 1`, so `zip(order, ranks)` pairs each index with itself plus
one and the sum reduces to `sum(i + 1 for i in positives)` — a function of **where the
positive labels sit in the input array** and nothing else. MEASURED: it returns 1.0 for
a perfect ranking, 1.0 for a perfectly inverted one, and 1.0 for a random one.

Replaced by `slopdet.metrics.auroc`, tie-corrected and checked against
`sklearn.metrics.roc_auc_score` on six cases including all-ties. `tests/test_metrics.py`
holds the regression guard. The consequences for the v1 post-mortem are in that section.

### 9.2 Paired rows straddled the train/eval split

The split shuffled rows and cut per `(register, label)` with no reference to
`split_hint`, and `slopdet.pairs` only recognised 3 of the 5 pair prefixes. On the
shipped v2 parquet that put **4,634 of 22,176 eval rows (21%) in the same pair family
as a training row** — a human source in `writingprompts` training the model that then
scores its rewrite in `rewrite_pair`. `slopdet.pairs.split_rows` now assigns whole
families and MEASURED 0 straddling families on the same corpus, with every register
still landing between 4.8% and 5.8% in val and cal.

The same fix raised the joinable contrastive pairs from 15,812 (of which 361 came from
a text-prefix fallback that could match unrelated documents) to 25,424, all from
explicit keys: `para:` and `premise:` were previously invisible to the pairing code
despite `scripts/merge_all_gen.py` documenting them as the contract.

### 9.3 v2 training aborted on its own registers

`register_allowed` had no entry for `rewrite_pair` or `respond_pair`, so the K3 schema
gate rejected 12,803 rows and training could not start on the shipped corpus at all.
`ALLOWED_REGISTER_SUFFIXES = ("_pair",)` covers the `<method>_pair` names
`scripts/merge_all_gen.py` generates.

### 9.4 One model definition instead of three

`DetectorBundle` existed in the exporter, the evaluator, and again as
`EncoderDetector` in the trainer, with **different token-head widths** (`n_lanes` vs
`n_lanes + 1`) and hardcoded lane counts of 8 and 4. A v2 checkpoint (4 lanes) loaded
into any of them under `strict=False` and scored with a partly random head.
`slopdet.detector` is now the only definition, and `from_checkpoint` reads the lane
count off the token head and refuses to load a checkpoint with missing tensors.

### 9.5 Module layout

| module | holds |
|---|---|
| `slopdet.corpus` | corpus resolution (local → HF), row loading, schema guards |
| `slopdet.pairs` | pair keys, families, the pair-safe split, contrastive pairs |
| `slopdet.metrics` | AUROC, TPR@FPR, per-register metrics, cross-register matrix |
| `slopdet.detector` | the model, and loading a checkpoint into it |

Scripts import these instead of each other. `scripts/train_cpu_scorer.py` no longer
reaches into `notebooks/`, `scripts/eval_trained.py` no longer imports the trainer, and
`scripts/eval_proper.py` is gone — folded into `eval_trained.py`, which is the single
canonical post-training step.

Removed as dead: `notebooks/` (Colab path abandoned), `dashboard.html` (one-off,
referenced nowhere), `docs/TRAINING_PREP.md` (v1 Colab prep pointing at the deleted
notebook), and `slopdet.{jlens,teacher,heads,student}` (the distillation design §3.1
superseded; zero importers).
