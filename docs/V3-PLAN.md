# ITAIS v3 — the full plan: dataset → span model → why-layer → shipped bundle

**Written 2026-08-25.** Plan only; nothing here is built until its phase says so. Supersedes the
dataset/eval sections of `docs/HANDOFF.md` where they conflict; the handoff's invariants (verbatim
quotes, copy rules, per-register thresholds) all still bind.

Fact classes: MEASURED / CITED / ESTIMATE / CONJECTURED, as in the handoff.

**Why v3 exists (MEASURED, `artifacts/corpus_probes.json`, 2026-08-25):** a register-tag-only
classifier scores **0.989 AUROC** on v2; length alone scores 0.705; 81% of docs live in single-label
registers. Any model trained on v2 can look great without reading the writing. v3 is the corpus that
makes the numbers mean something, plus the span model that makes the product do what ZeroGPT does —
mark *which part* is AI — with a why-layer that is evidence instead of vibes.

---

## 0. Product contract (what inference returns)

Input: selected text, any length. Output, always the same shape:

```json
{
  "doc":   {"matches_ai_pile": 0.87, "register": "blog", "threshold": 0.62, "verdict": "above"},
  "spans": [
    {"start": 0,   "end": 214, "lean": "human", "score": 0.08},
    {"start": 214, "end": 611, "lean": "ai",    "score": 0.91,
     "why": [
       {"kind": "pattern", "id": "glue.leverage", "quote": "leverage robust pipelines", "ratio": 41.2},
       {"kind": "model",   "id": "frames.recap",  "quote": "In conclusion, it's important to"},
       {"kind": "stat",    "id": "uniform_sentences", "detail": "9 sentences, len σ=1.8 words"}
     ]}
  ],
  "refusal": null
}
```

- **Span marking is the product** (the ZeroGPT behaviour): contiguous regions with a lean and a
  score, not one blended number. Doc verdict is derived from spans + doc head, never shown alone.
- **Why comes in three kinds, in trust order.** `pattern` = deterministic ontology hit, verbatim
  quote + measured overrepresentation ratio from our own corpus stats (shipped as a table). `model` =
  the token head predicts a *named lane* at that position; shown as "reads like X", cross-checkable
  because the lane vocabulary is the same 415-id ontology. `stat` = burstiness/uniformity-style
  document statistics, deterministic. An LLM never writes the explanation.
- **Refusal**: under 50 words → `refusal: "too_short"`, style hits only, no doc verdict (<100 words:
  verdict shown with a wide-uncertainty flag). Copy rules from the handoff hold everywhere: never
  "% AI", never "written by ChatGPT", never an authorship claim — `matches_ai_pile` against a named
  reference pile, enforced by `FORBIDDEN_SUBSTRINGS` tests.

## 1. Model and the quantization question

**Backbone: `LiquidAI/LFM2.5-Encoder-230M`** (bidirectional, 8k context, hidden 1024), loaded through
`slopdet.lfm.load_encoder_body` (the `AutoModel` random-weights trap is MEASURED and guarded). The
350M sibling buys +1.7 on Liquid's 17-task mean for +54% params; 230M is the right default, 350M is
the fallback if the span head plateaus (CITED, encoder model card, 2026-08-18).

**On "use their quantized weights": not available for encoders, and not how training works anyway.**
CITED from Liquid's model-format matrix (docs.liquid.ai, checked 2026-08-25): the encoder ships
safetensors only — GGUF ✗, MLX ✗, ONNX ✗. Every quantized artifact Liquid publishes (`-GGUF`,
`-ONNX` Q4/Q8, MLX 8bit, the QAD Q4_0 checkpoints) is for the **decoder** chat models, including the
confusingly-named `LFM2.5-230M`, which is a different model from `LFM2.5-Encoder-230M`. And you
can't fine-tune INT4/INT8 weights directly — quantized formats are inference containers. The plan:

1. **Train** in bf16 on a vast.ai box — 3090 (24 GB) or 4070 Ti Super (16 GB), both Ampere+ so bf16
   and FA2 just work (resolves O1). Full fine-tune, no LoRA: ESTIMATE ~3.7 GB for weights+grads+Adam
   at 230M, leaving room for batch 48–64 @512 tok on the 3090, 32–48 on the 4070 TiS. A weekend run
   is ~$10–20 of vast time (ESTIMATE).
2. **Quantize after training**: ONNX INT8 dynamic. Pipeline already proven on this exact body —
   MEASURED: 230.4 MB, 371 ms/512 tok on 4 vCPU threads (`artifacts/export_probe.json`).
3. **Gate**: quantized vs fp32 AUROC delta ≤ 1 point on every eval slice. If it fails, escalate in
   order: static QDQ INT8 with a calibration set → INT8-weights/fp32-activations → QAT (what Liquid
   calls QAD for their decoders — we'd do the same trick ourselves: short fine-tune with fake-quant).

## 2. Dataset v3

### 2.1 Fix v2 first (mechanical, CPU-only, no new data)

MEASURED failures → fixes, in order:

| # | Fix | Effect |
|---|---|---|
| F1 | `fetch_m4`: cap per **(domain, label)**, not per domain — the 8000-cap filled from a label-ordered file and produced five 100%-AI registers | ~38k human docs recovered; 5 registers become mixed |
| F2 | Merge register-name twins: `wiki_intro`+`wiki_intro_gpt`, each `hc3_x`+`hc3_x_gpt` → one register, two labels | kills the tag shortcut; per-register AUROC becomes defined |
| F3 | Length-match: window/truncate to a shared per-register length distribution (human 395 vs AI 191 median words is 0.705 AUROC by itself) | kills the length shortcut |
| F4 | Every row gets `era ∈ {pre, post}`; keep a post-2022 verified-human FPR slice out of training | makes calendar-reading visible |

**Gate G1:** `scripts/corpus_probes.py` passes — register-only ≤ 0.55 (two-sided), length-only ≤ 0.55
— before any generation money is spent. Probes to add: topic-only (TF-IDF, style words stripped,
≤ 0.65) and format-only (punctuation/markdown/casing, ≤ 0.60).

### 2.2 Generation campaign (the new slop)

Everything is **seed-paired**: each generation starts from a human doc already in the corpus and
carries its `split_hint` pair-key, so pairs never straddle splits and topic/length are matched by
construction. Provenance labeling only (we watched the machine write it); the existing blind-judge
contamination audit runs over every new slice.

**Axes.** Registers (from the human side, 9): fiction/WP, blogs, news, wiki, QA/eli5, academic
abstracts, how-to, reviews, email-ish. Generators (≥6, recorded per row): the gateway models already
wired (gemma-4-26b, laguna-s-2.1, ox-alpha-free) + 3–4 current families through the Command Code
gateway; hold out ≥2 whole families for the unseen-generator slice. Methods (7):

| Method | Prompt shape | What it teaches |
|---|---|---|
| `continue` | human prefix (10–50% of doc) → model continues | the classic hybrid; boundary label free |
| `rewrite` | full human doc → paraphrase | detection under paraphrase |
| `respond` | prompt/title → fresh generation | pure-AI baseline |
| `polish` | human doc → "fix grammar, improve flow" | the real-world gray zone |
| `expand` | human outline/first sentences → full doc | AI-bulk-from-human-seed |
| `antislop` | `respond`/`rewrite` + the 415 ontology ids as a **negative** instruction + "vary sentence length, be concrete" | AI-clean: the hard positive that defeats lexicon-only detection |
| `humanize` | AI output → second model "make this undetectable" | the adversarial case users will actually run |

Decoding matters (CITED, RAID: sampling changes flip detectors): generate each cell at two settings,
T≈0.7 default and T=1.0 + repetition penalty; record `decoding` per row.

**Volumes (ESTIMATE).** ~2,000 docs per register × 9 registers ≈ 18k per method across the 7 methods
≈ **~125k new AI docs**, generator-balanced. At ~700 tokens avg output that's ~90M output tokens —
request-count-bound on the existing gateways, so batch by register and let it run for days; nothing
GPU-shaped is blocked on it. Priority order if the budget squeezes: `continue` and `antislop` first
(they feed the span head and the hard-positive cell), then `humanize`, `polish`, the rest.

**Human-side additions** (from `docs/HUMAN-CORPUS.md`): human-slop from C4-2019 (marketing/SEO/press,
ODC-BY) ~20k docs; Stack Exchange + pre-2022 Wikipedia revisions for timestamped bulk; ICNALE +
post-era human as **eval-only**. Trainable vs eval-only licence pools stay separate in the manifest.

### 2.3 Span supervision (what makes it ZeroGPT-shaped)

Exact boundaries come free from construction, never from a judge model:

- `continue` docs: boundary = end of human prefix. Vary the cut 10–90%, mid-sentence cuts included
  (CITED, SemEval-2024 Task 8C construction).
- **Stitched hybrids**: interleave sentences/paragraphs from a pair-keyed (human, AI) doc pair —
  1–3 switches per doc, segments ≥ 2 sentences. ~30k docs assembled from existing + new pairs.
- `polish`/`humanize` docs: diff the output against the source; changed regions ≥ N chars become
  AI-labeled spans, unchanged regions keep the source label. Cheap, surprisingly precise, and it is
  exactly the mixed-authorship case real users paste in.
- Every span label carries provenance in the parquet (`spans` column already exists in v2's schema).

**Gate G2:** re-run probes on v3 including a new one — *boundary-position-only* (predict span label
from position in doc alone, must be ≈ 0.5, catches "AI is always the second half" artifacts). Plus
the standing rule: a corpus build ships with `artifacts/corpus_probes.json` attached or it doesn't ship.

## 3. Training recipe

One body, three heads, multi-task from the start (the two-head trainer in `scripts/fine_tune_lfm.py`
grows a third):

- **Doc head**: 2-class, mean-pooled. Loss weight 1.0.
- **Token head**: 3-class per token — human / ai / ai-lane-overlay. Concretely: binary
  human-vs-ai per token (the span signal), plus the existing lane vocabulary as a secondary softmax
  so a token can be "ai + reads-like-glue". Loss weight 0.5, `ignore_index` on padding.
- **Boundary head** (CONJECTURED, ablate): per-token "a switch happens here", label-smoothed ±1
  sentence. Cheap, and segmentation quality is the product, so it earns its ablation slot.

Curriculum: epoch 0 on pure docs only (doc head warms up without noisy span gradients), then mixed.
Windowing: 512 tokens, stride 384; doc label broadcast to windows, span labels clipped per window;
inference aggregates by max-pool over window scores per sentence. Class imbalance: sample so each
batch is ≈50/50 doc-label and ≥25% docs containing a boundary. Hard-negative mining after round 1:
score the full human pile, take the top-FPR 5k docs and the `antislop`/`humanize` false negatives,
upweight 3× in round 2. Seeds, git sha, data sha256 in the manifest per run; checkpoint + HF push
every 1000 steps (already wired: `c95f445`).

ESTIMATE compute: ~350k docs → ~500k windows/epoch → 2–4 h/epoch bf16 on the 3090, a bit more on the
4070 TiS, 2 epochs + a mining round ≈ a weekend of box time. Nothing here needs more than one card.

## 4. Segmentation + why-layer (inference-side design)

- Token scores → sentence scores (mean over tokens) → **hysteresis segmentation**: enter AI-span at
  sentence score > τ_hi, exit at < τ_lo, minimum span 2 sentences. Kills the flicker that makes
  per-token heatmaps unusable. τ per register from calibration.
- Each AI span's `why` is assembled, not generated: ontology hits inside the span (quote + ratio from
  the shipped overrepresentation table, computed once from v3 stats); token-head lanes that fired
  (agreement with a regex hit upgrades both to "strong"); doc-stat flags (sentence-length σ, em-dash
  rate, tricolon count) computed deterministically.
- Human-leaning spans get why too (`why_human`: anchors, numbers, dates, contractions — already in
  `slopdet.explain`), so the UI never implies "no evidence = human".

## 5. Evaluation — what "better" means here

Per register × per slice, every run, all in the eval report:

| Slice | Metric | Gate to ship |
|---|---|---|
| in-register mixed (coai, raid, F1/F2-fixed m4, hc3, wiki) | AUROC, TPR@1%FPR | beats CPU floor (0.8643 AUC, MEASURED — and re-measure it on v3 since coai_test leaked) |
| unseen-generator holdout (exists) | AUROC, TPR@1%FPR | ≥ 0.85 AUROC (ESTIMATE, tighten after run 1) |
| unseen-register (hold out one whole register) | AUROC | report; no gate v1 |
| **span quality** on stitched+continue holdout | boundary MAE (sentences), span IoU, per-sentence AUROC | MAE ≤ 1.5 sentences, IoU ≥ 0.7 (ESTIMATE) |
| paraphrase/humanize attack | TPR@1%FPR | report; expect degradation, measure it honestly |
| post-era verified human | FPR at operating thresholds | ≤ 2× the pre-era FPR |
| ICNALE (L2 English) | FPR | ≤ 2× native FPR, else thresholds move |
| short text <50 words | refusal rate | 100% refusal by contract |
| INT8 vs fp32 | AUROC delta per slice | ≤ 1 point |

Baselines in every report: the CPU logistic floor, and the deterministic ontology lane alone — if the
neural spans don't beat regex spans on IoU, we ship regex and say so (the handoff's stop-condition,
still binding).

## 6. Phone deployment (the actual requirement)

The detector runs **on the phone**, not on a server. Constraints and the plan that satisfies them:

- **Runtime**: ONNX Runtime Mobile, XNNPACK EP, mmap'd model load. One artifact serves Android, iOS
  (CPU/CoreML EP) and the desktop eval harness, which is why ORT beat ExecuTorch here. Inference is
  burst-shaped (score a selection, sleep), so thermals and battery are non-issues; no sustained load.
- **Size budget: ≤250 MB** in the app. INT8 dynamic already lands at 230.4 MB (MEASURED). If the app
  needs smaller, the ladder is: prune the 65k multilingual vocab to the English subset actually seen
  in training (embedding is 67M of the 230M params; CONJECTURED saving ~40–50 MB, needs a tokenizer
  round-trip check) → Q4 weights via ORT MatMulNBits (~130 MB, ESTIMATE) — every rung gated on the
  same ≤1pt AUROC rule. Ship over Wi-Fi as a post-install download, not inside the APK.
- **Latency budget (default target until O3 names a device): mid-range Android, Snapdragon
  7-series/8 GB.** Gates: ≤700 ms per 512-token window on that class of device, cold load ≤2 s,
  a typical selection (≤3 windows) fully scored ≤2 s. The 371 ms/512-tok on 4 vCPU threads (MEASURED)
  suggests flagships will be comfortably under; the mid-range number must be measured on-device in
  phase Q, not extrapolated.
- **Everything else in the bundle is tiny**: ontology + ratios + calibration + tokenizer ≈ a few MB.
  The why-layer costs nothing at inference beyond regex over the selection.

## 7. Shipped bundle

`artifacts/itais-v3/`: `detector.int8.onnx` + tokenizer + `calibration.json` (τ_hi/τ_lo + doc
threshold per register) + `ontology_ratios.json` (pattern → overrepresentation ratio) + `manifest.json`
(git sha, data sha256s, per-slice metrics, trained_on/never_trained_on, licence notes incl. LFM Open
License v1.0) + model card with the copy rules and known failure modes (attacked text, post-era FPR,
L2 numbers) stated as measured.

## 8. Execution order

| Phase | Work | Depends on | Gate |
|---|---|---|---|
| D0 | F1–F4 v2 fixes, probe extensions | — | **G1** probes pass |
| D1 | generation campaign (§2.2), C4 human-slop pull | G1, gateway keys | contamination audit + **G2** |
| D2 | hybrid stitching + span labels (§2.3) | D1 partial | G2 boundary probe |
| T1 | 3-head training, 2 epochs + mining round (§3) | D0 minimum; better after D2 | beats CPU floor on mixed slices |
| T2 | ablations: boundary head on/off, 350M vs 230M if plateau | T1 | each component earns its keep or dies |
| Q | INT8 export + AUROC-delta gate (§1) + on-device latency check (§6) | T1 | ≤ 1 pt; §6 budgets |
| E | full eval report + model card + bundle (§5–6) | Q | every number sliced, classed, and in the manifest |

D1 runs unattended for days and blocks nothing: T1 can start on D0-fixed data and retrain when D1/D2
land. First trainable milestone is D0+T1 alone — that already answers "does the encoder beat the
floor on an honest corpus", which is the question v2 couldn't ask.

## 9. Open items

- ~~O1~~ **resolved 2026-08-25**: vast.ai 3090 / 4070 Ti Super — Ampere+, bf16, batch sizes in §3.
- **O2**: gateway request budget for ~125k generations → decides whether the method matrix runs full or in the priority order above.
- **O3**: exact target phone still unnamed → §6 sets a mid-range-Android default budget; naming a
  device only tightens the numbers, it doesn't block any phase before Q.
- **O4**: PERSUADE licence conflict (CC BY vs BY-NC-SA) — eval-only until resolved.
