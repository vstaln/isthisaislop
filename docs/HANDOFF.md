# HANDOFF — ITAIS: train the detector

**Revision:** 2026-08-18 (rev 2) · **Authoritative.** The two files under `docs/superpowers/plans/`
are historical: they specify a roberta-base doc classifier that this revision supersedes.
**Expires 2026-09-18** — after that, re-derive from `git log`, `artifacts/MANIFEST.json` and the
rebuild commands in §4 rather than trusting the STATUS block.

**Fact classes used below:** MEASURED (a command in this repo produced it — command named) ·
CITED (external source + date checked) · ESTIMATE (arithmetic, not run) · CONJECTURED (belief, with
the experiment that would settle it) · STALE-BY (expires on a named condition).

---

## STATUS

| | |
|---|---|
| Shipping today | deterministic lanes (415 ontology ids + construction + storyscope), CPU logistic `matches_ai_pile`, span stitch, calibration record |
| Neural model | none trained. Trainer exists and is smoke-tested: `scripts/fine_tune_lfm.py` |
| Export path | **gate passed** (§3.3): LFM2 encoder → ONNX → INT8, 230.4 MB, MEASURED |
| Data | built, **local-only** — `data/` is gitignored and absent from a fresh clone (MEASURED: `ls data` fails after `git clone`) |
| Baseline number to beat | unknown — `artifacts/sklearn_bundle.json` is gitignored, so its coai AUC is not recorded anywhere in the repo |
| Blockers | (B1) data parquets exist on one machine only; (B2) target phone/chipset for the on-device latency budget |

## NEXT ACTIONS (in order; each ends in a committed artifact)

1. **Record the floor.** Run the existing CPU scorer on the coai test slice and commit the number to
   `artifacts/MANIFEST.json`. Until this exists, no neural result can be called an improvement.
   `uv run python scripts/train_cpu_scorer.py --eval-only --report`
2. **Train the encoder doc+span model** (§3.1–3.2), on the T4:
   `uv run python scripts/fine_tune_lfm.py --arch encoder --model LiquidAI/LFM2.5-Encoder-230M --max-len 512`
   (validate the plumbing first with `--smoke`, which needs no corpus: MEASURED, 3 steps on CPU in 18 s)
3. **Evaluate per register, not globally** (§5). A single coai number is not a result.
4. **Re-run the export gate on the trained weights** and record the INT8 AUROC delta, not just the
   logit drift: `uv run python scripts/export_onnx.py --arch encoder`

## STOP CONDITIONS

- No AUC without the register slice it was measured on and the FPR the threshold was set at.
- No claim that the neural model is better until it beats the §NEXT-1 floor on the **fiction and blog**
  slices, not only on coai.
- Copy rules are load-bearing product behaviour, not style: never "% AI", never "AI-generated", never
  "written by ChatGPT", never an authorship claim. Enforced by `slopdet.report.FORBIDDEN_SUBSTRINGS`
  + tests (MEASURED: `python -m pytest -q` → 31 passed).
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
loses the token head) → roberta-base span model, which exports trivially and already has a trainer in
`notebooks/colab_pipeline.py`.

## 4. Data assets — and how to rebuild them

`data/` is gitignored, so this table describes state that exists on **one machine**. Every row needs a
rebuild command; that is what makes the table a handoff instead of an inventory.

| Path | Contents | Rebuild |
|---|---|---|
| `data/coai_train.parquet` | 62,460 docs (31,230 AI / 31,230 human), cols `text,label,model_name` | `colab_pipeline.download_coai` |
| `data/training/spans_coai_train.parquet` | 62,460 rows, cols `text,pile,slop_tags,human_tags,spans`; 172,169 spans, 0 non-verbatim | `scripts/label_coai_batch.py` (~3 min, 8 cores) |
| `data/training/labeled.parquet` | 250 eval docs (130 AI / 120 human), LLM lean + local spans | `scripts/export_training.py` from `eval/labels/*.jsonl` (committed) |
| `data/raw/storyscope/` | 5 AI stories per prompt (GPT-5.4, DeepSeek V3.2, Kimi K2.5, Gemini 3 Flash, Claude 4.6); human side excluded for copyright | `scripts/fetch_datasets.py --job storyscope` |
| `data/raw/gutenberg_fiction/` | ~393k public-domain chunks, human fiction | `--job gutenberg_fiction` |
| `data/raw/blogs/` | 19,320 Schler blog files — **raw HTML/XML, needs text extraction** | `--job blogs` |
| `data/raw/writingprompts/` | modern human short fiction (~600 MB) | `--job writingprompts` |
| `data/raw/scp/` | CC-BY-SA modern fiction (~116 MB) | `--job scp` |

**Fix owed here:** `artifacts/MANIFEST.json` is the one artifact path git tracks — put row counts and
sha256 per file in it, so a second machine can prove it rebuilt the same corpus.

**Register coverage is the real data problem** — `docs/HUMAN-CORPUS.md` specs the fix. coai is arxiv abstracts vs LLM paraphrases: academic
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
- Colab free T4: no bf16, no flash-attn 2, session death is normal → checkpoint to Drive every ~500
  steps and make resume idempotent; stage shards to local disk first, Drive I/O will dominate otherwise.
- LFM2 encoders need `trust_remote_code=True`, and `AutoModel` silently hands back random weights —
  always go through `slopdet.lfm.load_encoder_body` (§3.2).
- `scripts/fine_tune_lfm.py --smoke` runs the whole training path on synthetic data with no corpus
  (MEASURED: 3 steps, 18 s, CPU) — use it to validate edits before spending a Colab session.
- Tests: `python -m pytest -q` → 31 pass (MEASURED, this revision).

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
