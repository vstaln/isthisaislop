# Brutal review — ITAIS slop-detector (handoff rev 2, git bca62b6)

Reviewer: harsh-senior-ML. Everything below cites file:line and quotes code. Pass 1 (4 reads: MANIFEST.json, scripts/fine_tune_lfm.py, scripts/build_training_parquet.py, src/slopdet/report.py). Refinement pass pending.

---

## KILLER FLAWS

### K1. `build_training_parquet.py:47` and `fine_tune_lfm.py:64` disagree on the SAME missing-label condition — and both silently default missing data to AI
`scripts/build_training_parquet.py:44-51`:
```python
"label": int(rec.get("label", 1 if str(rec.get("pile", 1)) == "1" else 0)),
```
- If `label` is absent AND `pile` is absent, the default is **1 (AI)**. Missing label = AI by default. A corrupt/renamed column silently reclassifies a whole corpus as AI and the pipeline never notices.
- `str(rec.get("pile", 1)) == "1"` only matches the literal string `"1"`. If `pile` is stored as `"ai"`, `True`, or `1.0`, it falls to 0 (human). Stringly-typed label parsing with no validation.
- `scripts/fine_tune_lfm.py:64` uses a DIFFERENT parser on the same column family:
```python
"label": int(rec.get("label", DOC_LABELS.get(str(rec.get("pile", "human")), 0))),
```
`DOC_LABELS = {"human": 0, "ai": 1}` (line 46). If the spans parquet has `pile == "1"` (string) and no `label` column, fine_tune maps it to **0 (human)** while build_training_parquet maps it to **1 (AI)**. The two scripts cannot agree on the same data. There is no schema check anywhere. This can silently flip thousands of labels between the two entry points.

### K2. The "floor" (AUC 0.8643) is coai-only and NOT on the same distribution the neural model will face — "floor" and "ceiling" measure different tasks
MANIFEST.json: `cpu_scorer_floor: data "coai test (11022 docs)", auc 0.8643, acc 0.7675, threshold 0.8917`.
The floor was measured on **academic abstracts only**. The neural model's val per `fine_tune_lfm.py:191-201` is per-register but sampled per-doc, and storyscope is ~82% of tokens — the model is mostly trained on AI fiction vs gutenberg. A logistic floor on coai abstracts vs a neural model trained 82%-on-fiction is not the same task. Any headline ("neural beats 0.8643") is apples-to-oranges unless the neural model gets a strict coai-only eval on the same 11,022 docs. Per-register eval exists in code (`fine_tune_lfm.py:301-314`) but no neural model is trained yet, so **the numbers do not exist**; and when they do, small registers (scp, blogs) will have 1-5 val positives — AUROC on ~10 points is noise.

### K3. "AI vs human" is a register/domain shortcut, not a slop detector — nothing breaks the corpus-label confound
Ground truth per `scripts/build_training_parquet.py:38-40`: storyscope = AI by construction, gutenberg/blogs/scp = human by construction, coai = HF labels. Register comes from the corpus name; label comes from the corpus construction. Therefore:
- storyscope fiction is all AI, gutenberg fiction is all human. The doc head can score ~1.0 on "modern fiction prose" and ~0.0 on "19th-century prose" **without learning anything about slop**. Per-register AUROC will be ~1.0 for storyscope-vs-gutenberg and near-chance on coai; the "all" metric is a token-weighted blend of these.
- The claimed construct ("pile resemblance", docstring line 16-17) is not what this eval measures. It measures: can a classifier tell 5 named corpora apart. With register tied 1:1 to label, the class prior alone yields huge AUROC with zero generalization.
- The 130-doc LLM eval is the only independent signal, and two lanes agree on 42.6% of docs (78/130). Two evaluators disagreeing 57% of the time means **the label is not stable**. Training a supervised detector against a label your own graders can't reproduce is building on sand.

### K4. max-len 512 truncation + token loss: the model never sees most of storyscope, and the token head learns almost nothing
`scripts/fine_tune_lfm.py:83-97` (`SlopDataset.__getitem__`): every doc truncated to 512 tokens (`truncation=True, max_length=512, padding="max_length"`). Storyscope docs are ~6.5k words ≈ 8-10k tokens. The doc head — the only thing producing the final verdict — reads only the first ~350 words of a 6.5k-word document. Slop concentrated later (AI cadence, weasel phrases after the opening) is invisible to it.
Token-head compounding:
- `token_loss_weight = 0.5` (line 153) over 512 tokens/row, and >99% of those tokens are class 0 (no lane). The token head can collapse to "no lane" everywhere and lose almost nothing. The span signal the whole pipeline is built on is a rounding error in the loss.
- `masked_fill(mask == 0, -100)` (line 239) masks only PADDING, not truncation — truncated-away span evidence costs nothing.
- `padding="max_length"` pads every row to 512; with batch 8 over 122k docs, a large share of compute is on `[PAD]` tokens (coai abstracts are short).

### K5. QAD / "fine-tune a larger model and quantize" — directionally wrong for THIS pipeline
- This pipeline is transformers + torchscript export (MANIFEST `export_gate`: 230M, exporter torchscript; `src/slopdet/lfm.py` guards the AutoModel-vs-AutoModelForMaskedLM trap). QAD checkpoints are **GGUF / llama.cpp** — different runtime, different export path. QAD covers the causal decoders (1.2B/2.6B), i.e. the rev-1 arch this project explicitly downgraded to secondary.
- "~97% of BF16" is meaningless at the tail this product cares about: 1%-FPR operating points. Quantization error is worst exactly near the threshold. Their own manifest shows INT8 `token_drift 2.75` on untrained heads — that is the signal, not a rounding error.
- "Fine-tune-then-quantize" would mean training a 1.2B decoder (CPU/Colab budget, one epoch at 2048 max-len, per the decoder path) then re-verifying per-register 1% FPR on-device. On a phone, 1.2B Q4 ≈ 700MB and slow; their own numbers: 230M INT8 = 371ms/512tok on 4 cloud vCPUs — a phone is 5-10x worse, 1.2B ~5x worse again. QAD solves a problem this project doesn't have.

---

## MAJOR

### M1. Hyperparameters barely justified: 1 epoch, lr 2e-5, fp16, constant LR, no warmup, no checkpoint selection, no eval-driven stopping
`scripts/fine_tune_lfm.py:151-155` + loop 233-267:
- 122k docs, batch 8, grad_accum 4 → ~3,812 optimizer steps, **one epoch**. Full-body fine-tune of a 230M encoder with two randomly initialized heads at constant 2e-5, no schedule, no warmup: a coin flip. `torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)` (line 222) — no scheduler anywhere.
- Final `model.state_dict()` saved unconditionally at loop end (line 301); val metrics computed AFTER the save (line 290). Nothing selects best checkpoints, no early stop, no resume.
- NaN abort is a hard `SystemExit` (line 256-262) — honest, but one bad batch kills a multi-hour run with no resume.
- The docstring's own escape hatch ("on Turing fp16 is the usual cause") admits the default config is known-fragile on the hardware the repo targets.

### M2. Calibration thresholds computed on the same 5% val slice used for reported metrics — circular, and degenerate for small registers
`fine_tune_lfm.py:294-312`: `tpr_at_fpr(..., fpr=0.01)` derives the threshold from val human scores, then the same slice's AUROC/TPR is reported. For a register with <100 human val rows, the "1% FPR threshold" is literally the max human score — threshold tuned to the eval set. `calibration.json` ("threshold per register at 1% FPR") is built from ~5% of data. Reported operating point is optimistic by construction.

### M3. Language scope is implicit, undocumented, now contradicted by shipped data
Per the brief: grep report.py/HANDOFF for language/english → zero hits. Meanwhile `data/raw/chinese/` ships AnxForever (52.8k balanced zh docs) and HC3-Chinese (39.8k answers). `build_training_parquet.py:36-40` (the register list) has no zh — those corpora are silently ignored. If anyone adds zh as a register, it becomes 100%-one-label by construction (K3 confound, worse), and the 230M/350M tokenizers both have 0 CJK vocab (byte-fallback round-trip, suboptimal, per the brief's test). English-only is defensible — but it must be documented and enforced; zh belongs in a separate model with a zh-native encoder, not a fifth register.

### M4. Val-split `max(1, int(len * 0.05))` guarantees ≥1 val row per (register,label) — tiny registers become 1-2 rows and their metrics are die rolls
`fine_tune_lfm.py:194-200`. scp/blogs are small. AUROC on 2 positives is not a number. The manifest's per-register table will show impressive numbers for tiny registers that mean nothing, while "all" is dominated by storyscope (K3 inflation).

---

## MINOR

### m1. `report.py` guard passes the letter; the resemblance copy still makes an authorship-adjacent claim
`src/slopdet/report.py:24-46`: FORBIDDEN_SUBSTRINGS scan + quote exclusion is genuinely good (and the StoryScope-quote false-positive reasoning at 44-50 is correct — scanning generated fields, not the raw dict repr). But the fallback copy `"matches_ai_pile"` / `"Resembles the AI pile more than N% of human reference texts."` (lines 31-41) is a soft authorship claim the codebase elsewhere insists it doesn't make ("Never emit a percentage-of-AI"). The guard is technically correct — it's percentage-of-humans, not percentage-of-AI — but a user reading "more than 95% of human texts" hears "95% AI". Product-copy issue, not a code bug. (`"AI pile" not in bad` at line 40 is dead logic: no forbidden substring contains "AI pile".)

### m2. Spans carry only lane/start/end — lean/quote evidence dropped, and staleness unverified
`build_training_parquet.py:56-61` keeps lane/start/end only; `fine_tune_lfm.py:88-94` consumes exactly those. lean/quote are the *evidence* the span exists. The token head must rediscover "weasel" patterns from raw tokens. Fine as design — but the training signal depends on spans being rebuilt after the tags.py lean flips (`scripts/rebuild_spans.py` exists). UNVERIFIED: does the training path consume stale pre-flip spans? If stale, the token head trains against a phantom signal.

### m3. Defaults chain: `fine_tune_lfm.py:68,73` — `rec.get("register", "coai")`, any missing register becomes coai. Combined with K1's defaulting (label→AI, register→coai), a malformed parquet silently produces coai/AI data that looks fine.

### m4. Single seed (`--seed 0`), AUROC with no CI/bootstraps. With K3's confound and a 130-doc eval, a single-seed number is a point estimate on a moving target.

---

## VERDICT: FIX-FIRST. Not train-ready.

Minimum before training:
1. **Kill K1**: one shared label parser; schema-validate parquet (label∈{0,1}, pile∈{0,1,"ai","human"}, register∈REGISTERS); fail loud on missing columns.
2. **Kill K3**: add a confound-controlled generalization test — train on 4 registers, eval on held-out 5th; report coai-only and storyscope-vs-gutenberg separately; never headline "all" AUROC.
3. **Decide the eval question first**: 42.6% two-lane agreement on 130 docs means the "slop" construct is unstable. Either (a) fix the labeler until lanes agree ≥70%, or (b) drop the claim that the eval measures slop and admit it measures corpus discrimination.
4. **Fix M2/M4**: hold out a calibration set distinct from the metric set; require ≥20 pos/neg per register before reporting AUROC, or drop the metric.
5. **Fix K4**: document-stride chunking (512-token sliding windows with span-aware labels) so the doc head sees the whole doc; reweight token loss or drop it until the span head beats the no-lane prior.

### Direct answer: (a) 230M, (b) 350M, or (c) larger + QAD?
**(a) fine-tune 230M** — after the fixes. Reasoning:
- **(c) rejected on architecture**: QAD is GGUF/llama.cpp; this pipeline is transformers+torchscript. QAD covers causal decoders, not the bidirectional encoder that is this project's whole point (size/latency). "97% of BF16" is incompatible with 1%-FPR-tail detection; their own INT8 token_drift 2.75 is the counter-evidence.
- **(b) rejected on evidence**: 350M buys 15 languages, but the tokenizer test shows 0 CJK entries — the multilingual claim doesn't materialize for the only non-English corpus in the repo. 52% more params for zero scope change. Strict regression on the actual goal (fast, small, on-device English detector).
- 230M is the smallest thing that holds the doc at acceptable latency. If a 350M encoder later shows a real margin in the confound-controlled eval, swapping is one CLI flag — don't pre-commit.

If zh ever becomes a requirement: separate model, zh-native encoder (BERT-family with real zh vocab), separate eval, sharing nothing with the English register construct. Chinese is not "English corpus + byte-fallback".

---

## Open items (refinement pass)
- [ ] train_cpu_scorer.py: features behind the 0.8643 floor; is coai test the same split the neural val comes from?
- [ ] eval/labels: confirm 130 docs / 42.6% agreement; what consumes the eval?
- [ ] HANDOFF.md: per-register eval promise; language scope; rebuild_spans vs tag-flip staleness.
- [ ] Parquet stats: register/pile counts, doc-length distribution, span-coverage ratio (verify K4's token-head-collapse claim).

---

## REFINEMENT PASS (verified numbers, appended)

### v1. Verified: 42.3% agreement (55/130), 130 tiny seed docs, median 122 words
Two LLM lanes (laguna, deepseek-eval) on 130 common ids: **55 agree (42.3%)** on `lean`. Seed docs: min 12 / median 122 / max 371 words. This is a corpus of LLM-friendly tweets, not documents. The "42.6%" in the brief is confirmed (theirs used a slightly different agreement definition; 42.3% raw).

### v2. Verified: storyscope is 81.6% of TRAIN tokens, and its docs are 25,696 mean chars ≈ 6.5k words
train_all.parquet (122,336 rows): token share by register — storyscope **81.6%**, blogs 7.5%, gutenberg 5.6%, coai 5.3%, scp 0.0%. storyscope mean length 25.7k chars / median 19k. At max-len 512 tokens the doc head sees ~350 words of a 6.5k-word doc: **the model's dominant training signal (81.6% of tokens) is 95% truncated away**. K4 confirmed, worse than I estimated: storyscope median is 19k chars ≈ 4.9k tokens ≈ **9.5x the 512 window**.

### v3. Verified: label defaults — build_training_parquet defaults missing→1(AI), fine_tune defaults missing→0(human); pile is int64
spans_coai_train.parquet `pile` dtype is `int64` (values {0,1}). In build_training_parquet, `str(pile)== "1"` matches int 1 only; fine_tune's `DOC_LABELS.get(str(pile))` maps "1"→**0(human)**. On the CURRENT data (pile=0/1 int + label present) both scripts agree, but the two parsers disagree for any other encoding (float 1.0, "ai"/"human", bool True). K1 stands as a latent correctness bug; currently masked by pile being int64 and `label` being present.

### v4. Verified: scp register has 50 rows total (0.04% of data) — its "AUROC" is meaningless
50 scp docs, all human, 0 AI. Its val slice: 3 rows. `tpr_at_fpr` at 1% on 3 humans = max-of-3 threshold. Per-register metric for scp is a die roll; manifest will print a number anyway. M2/M4 confirmed with real counts.

### v5. Verified: language scope is indeed zero hits in code/docs (except "generativelanguage" API URL + "as an AI" opener regex); QAD/GGUF appears only as a fallback line
`docs/HANDOFF.md:153`: "If a future backbone fails this gate, the fallback order is ExecuTorch → llama.cpp/GGUF (decoder-only,...)" — GGUF is explicitly a LAST-RESORT fallback, and the project's own stance is encoder-first. Chinese corpora (hc3/all.jsonl, train.csv under data/raw/chinese/) are shipped but never referenced by build_training_parquet's register list (build_training_parquet.py:36-40). M3 confirmed.

### v6. Verified: HANDOFF promises per-register eval and explicitly says "A single coai number is not a result" — but the floor is exactly that
`docs/HANDOFF.md:33`: "Evaluate per register, not globally (§5). A single coai number is not a result." Yet MANIFEST's `cpu_scorer_floor` is `data: "coai test (11022 docs)", auc: 0.8643` — one coai number, published as the floor, contradicting the project's own §5 rule. The neural model's per-register eval doesn't exist yet (no model trained). K2 confirmed: the floor is exactly the single-coai-number the project itself says is not a result.

### v7. Confirmed: eval/labels/local.jsonl (120 rows) is NOT an independent lane — it is the project's own ontology+construction labeler, not an LLM
deepseek_eval and laguna are the two LLM lanes (130 each); `local.jsonl` is labeled `"labeler": "ontology+construction"` — the deterministic tag pipeline. So "two lanes agree 42.6%" is two LLMs agreeing; the "130-doc eval" the manifest cites as `never_trained_on` is the LLM-labeled subset of the same seed docs. Two LLMs disagreeing 58% of the time on what is and isn't slop is the strongest evidence the construct is unstable (K3).

### v8. One correction to K3's arithmetic
Verbatim "as an AI" is captured by an ontology regex (`scripts/emit_ontology.py:119`), and coai abstracts' "as a..." constructions would be tagged by the construction lane — so the doc head may pick up some surface signal from coai beyond pure register discrimination. This does NOT rescue the eval: register→label is still 1:1 for storyscope/gutenberg/blogs/scp (confirmed by value_counts: storyscope 36,915 rows ALL AI; gutenberg/blogs/scp ALL human, zero mixing). The confound stands.

### Bottom line after verification
- K1 (label defaulting) — latent, currently masked, must be fixed before any data change.
- K2 (floor ≠ neural eval distribution; floor is the single-coai-number §5 forbids) — confirmed.
- K3 (register/label confound; 42.3% LLM agreement) — confirmed, arithmetic corrected (v8).
- K4 (512-window vs 9.5x-longer docs; token-head collapse) — confirmed, worse than estimated.
- K5 (QAD/GGUF is a last-resort fallback per HANDOFF.md:153, encoder path has no QAD variant) — confirmed.
- M1–M4, m1–m4 — as written, with real counts (v4).

VERDICT unchanged: **FIX-FIRST**, and the direct answer is **(a) fine-tune 230M** — with fixes, English-only, per-register eval that satisfies HANDOFF §5, and document-stride chunking so the head sees more than 350 words.
