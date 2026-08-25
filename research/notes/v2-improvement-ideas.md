# v2 improvement ideas — ranked

Ideation note, 2026. Basis: `research/notes/brutal-review.md` (verified refinement pass), `docs/HANDOFF.md` rev 2, `scripts/fine_tune_lfm.py` (data path + Muon/scheduler section), `src/slopdet/scorer.py`. Current-state items listed in the task brief are treated as done and NOT re-proposed.

Ranked by expected real-world detection impact ÷ implementation cost, under constraints: one 16GB GPU ($0.10–0.11/hr), solo maintainer, ≤10h runs, no new heavy deps. Adversarial rule applied: an idea earns "do in v2" only if its mechanism plausibly moves TPR@1%FPR on *unseen-generator or paraphrase* slices — the only slices that test the thing we actually sell. Most ML brainstorm ideas are noise; several below are explicitly marked as such.

---

## Tier 1 — do in v2

### 1. Same-prompt paired contrastive training (kill the register confound at the root)
**Mechanism.** The repo already owns the perfect asset and doesn't use it as pairs: StoryScope is 5 AI stories *per writing prompt*, and writingprompts is human stories *on the same prompts*. Join on prompt; sample pairs (AI-story, human-story) sharing a prompt into a contrastive term (simple: supervised loss on pairs + a margin/InfoNCE term pulling same-prompt cross-label pairs apart in pooled space). Optionally add gutenberg-modernized pairs later. This is the only cheap intervention that makes "label" statistically identifiable independent of "source corpus" — the exact K3 failure. Without something like this, every other improvement trains a better corpus classifier.
**Impact: HIGH.** Directly targets the #1 verified flaw; converts the unseen-generator holdout from a formality into a real generalization test.
**Cost:** ~4–6h (prompt-key join, pair sampler, one extra loss term). Risk: low-moderate — prompt keys must match; imperfect joins just yield fewer pairs, degrade gracefully.
**Verdict: DO IN V2.** Highest impact/cost ratio on the board.

### 2. Eval/calibration integrity package (institutionalize everything v1 got wrong)
**Mechanism.** One bundle of harness changes: (a) three-way split — train / calibration (thresholds live here) / reporting (metrics only, never touched by threshold selection), fixing M2's circularity; (b) checkpoint selection by unseen-generator + paraphrase holdout AUROC instead of "save last step"; (c) minimum-n gate: no per-register AUROC printed below 20 pos AND 20 neg (kills scp die-rolls, M4); (d) bootstrap CIs (1k resamples) on every headline number, ≥2 seeds for anything claimed; (e) the coai-vs-regex floor re-run restricted to the *same* doc set the neural model is evaluated on — same-distribution comparison or no comparison (K2).
**Impact: HIGH** — not because it raises AUROC but because it's the difference between knowing and believing you know. v1's numbers were mostly artifacts; this makes v2's numbers falsifiable.
**Cost:** ~3–5h, all harness code, zero training risk.
**Verdict: DO IN V2.** Non-negotiable prerequisite for claiming any of the other ideas worked.

### 3. Feature fusion: feed deterministic features into the doc head
**Mechanism.** The Feature-Augmented idea, minimal form: compute the existing `scorer.featurize` vector (ontology pattern counts + construction stats + length) per training doc, project it (small linear → layer norm), concatenate with the pooled encoder representation before the doc classifier. At inference the CPU scorer already runs in-process (`slopdet.scorer`), so serving adds one concat. The neural model gets the regex lane's evidence for free and learns residuals on top of it; the "beats the floor" claim becomes structurally honest because the floor is *inside* the model.
**Impact: MED-HIGH.** Deterministic lanes are the product's most trustworthy signal; fusing them usually helps most exactly at the 1%-FPR tail where the encoder alone is noisiest. Also gives a clean ablation: fused vs encoder-only quantifies what the body adds.
**Cost:** ~3–4h. Risk: low (feature vector can be zero-masked; smoke test covers graph).
**Verdict: DO IN V2.**

### 4. Token-loss rebalancing so the span head can't collapse
**Mechanism.** v1's token head could predict "no lane" everywhere at negligible cost (>99% class-0 tokens, weight 0.5). Fix: (a) reweight token CE by inverse class frequency or cap background weight (e.g., background:span ≈ 1:5 effective); (b) report span-token F1 per lane as a first-class metric, not just doc AUROC — a doc head with a dead span head is half a product since spans are the user-facing "why".
**Impact: MED-HIGH.** Spans + rubric are the product differentiator; a collapsed span head means the verdict has no evidence attached.
**Cost:** ~2–3h. Risk: low.
**Verdict: DO IN V2.**

### 5. Random-crop window sampling for long documents
**Mechanism.** At max-len 2048, storyscope docs (~9k tokens) still overflow ~4x. Instead of always taking tokens [0:2048], sample the crop start uniformly per epoch (or stride-sample 2 windows per long doc). Teaches position invariance, exposes mid/late-document cadence, and effectively multiplies long-doc training data at zero annotation cost.
**Impact: MED.** Cheap insurance against "slop concentrated after the opening" (the K4 mechanism, partially addressed by 2048 but not eliminated).
**Cost:** ~1–2h (dataset `__getitem__` change).
**Verdict: DO IN V2.**

### 6. Product verdict combiner: spans + rubric + doc score → one decision rule, with abstain
**Mechanism.** Define, in code and tests, how the two lanes merge: deterministic span density and rubric severity set a prior; the calibrated doc score adjusts it; output ∈ {likely-slop, likely-human, mixed, unclear}. Abstain ("unclear") fires on: <N words, doc score inside the calibration-set ambiguity band, or span evidence contradicting doc score. Thresholds selected on the calibration split at per-register target FPR (HANDOFF §5 already demands this — implement it as the actual product rule, not a manifest field). Report refusal rate as a metric, per HANDOFF.
**Impact: MED-HIGH for real-world quality.** Most user-facing errors will be threshold/combination errors, not model-capacity errors. Abstain converts the worst failure mode (confident wrong verdict on a 30-word selection) into an honest shrug.
**Cost:** ~4–6h including tests. Risk: low.
**Verdict: DO IN V2.**

## Tier 2 — strong candidates, sequence after Tier 1 results

### 7. Humanization/paraphrase attack augmentation
**Mechanism.** Take a slice of AI-positive docs, run them through 2–3 "make this sound human" rewrite passes (APIs already in use for the eval lanes), label them AI (provenance-preserving: content-derived, rewrite-of-AI). Add as training data and as a permanent holdout slice. Teaches invariance to surface humanization — the actual adversarial direction for a slop detector.
**Impact: HIGH if the threat model includes evasion (it should); MED otherwise.**
**Cost:** ~6–10h wall-clock incl. API calls and dedup; risk: moderate — rewrite labels inherit the "is a rewrite still slop?" construct question (the 42.3% lane-agreement problem in miniature). Mitigate by keeping these out of the primary loss weight until the paraphrase holdout shows they help rather than hurt.
**Verdict: DEFER TO V2B** unless the paraphrase holdout from the current mix already shows degradation — then promote.

### 8. Verdict-level stacking of CPU scorer + neural score
**Mechanism.** Train a tiny logistic stacker on the calibration split: features = CPU-scorer percentile, neural doc logit, span-hit count, register. Cheaper and more interpretable than feature fusion (#3), and keeps the CPU scorer shippable alone on devices too small for the encoder.
**Impact: MED.** Overlaps heavily with #3; stacking is the fallback if fusion destabilizes training.
**Cost:** ~2–3h.
**Verdict: DEFER TO V2B** — do #3 first; build this only if fusion underperforms or the small-device SKU matters.

### 9. Hard-example mining / focal weighting on the doc head
**Mechanism.** After a first pass, upsample docs the current model gets wrong (esp. human docs scored AI — the costly error at 1% FPR) for a short second phase; or use focal loss throughout.
**Impact: MED at best.** Classic mining helps most when data is abundant and clean; here label noise (see #12) means "hard examples" are often mislabeled ones — mining amplifies label noise as readily as signal.
**Cost:** ~3h.
**Verdict: DEFER TO V2B**, and only with a manual read of 50 mined examples first to confirm they're hard-not-wrong.

### 10. EMA of weights for evaluation/export
**Mechanism.** Keep an exponential moving average of weights (decay ~0.999) updated each step; evaluate and export the EMA copy. Nearly free, routinely worth a few tenths of AUROC points and visibly better tail calibration on short fine-tunes.
**Impact: LOW-MED but almost guaranteed positive; the rare free lunch.**
**Cost:** ~1–2h. Risk: negligible.
**Verdict: DO IN V2** (piggyback on Tier-1 training run; not worth its own headline).

### 11. Auxiliary register-prediction head (diagnostic first, adversarial later)
**Mechanism.** Plain version: add a register-classification head; if it's near-perfect from the doc representation while the unseen-generator holdout is weak, that *measures* how much capacity goes to register shortcuts. Gradient-reversal version: actively strip register information from the pooled vector.
**Impact: MED as diagnostic, UNPROVEN as treatment.** GRL on transformers is notoriously unstable and can degrade the main task; the paired-contrastive idea (#1) attacks the same confound with a supervision signal instead of an adversary, which is more reliable per hour spent.
**Cost:** diagnostic ~2h; GRL ~4–6h plus tuning pain.
**Verdict: DIAGNOSTIC DO IN V2 (cheap, informative); GRL DEFER TO V2B** and only if #1 leaves residual confound measurable on the holdouts.

## Rejected or deprioritized (say no early)

### 12. Label smoothing on the doc head
**Mechanism would be:** soften 0/1 targets to reduce overconfidence. **Why rejected:** our problem is not overconfidence on clean labels — it's that some labels are wrong (LLM lanes agreed 42.3%; provenance labels are corpus-level). Smoothing a wrong label doesn't make it right; it just hides miscalibration that the calibration split would have exposed. Fix label provenance instead (#1, #7). **REJECT for v2.**

### 13. Curriculum ordering (easy registers → hard)
**Why rejected:** curricula show reliable gains mainly in RL and noisy-label regimes; on 250k docs / ≤10h runs, the curriculum consumes the same compute budget as simply training longer, and "easy" here (coai abstracts) is exactly the distribution we're trying NOT to overfit to. Length-balanced mix + random crops (#5) capture most of the benefit with none of the scheduling code. **REJECT for v2.**

### 14. R-Drop consistency regularization
**Mechanism would be:** two dropout-noised forwards + KL penalty. **Why deferred:** doubles forward cost on the entire run (relevant at ≤10h budget), and its published wins are on GLUE-scale tasks with many epochs; we run ~1–2 epochs where dropout hasn't even converged to being a regularizer yet. **DEFER TO V2B** only if multi-epoch training becomes the norm.

### 15. Bigger backbone (350M) / new architectures
**Why rejected for v2:** HANDOFF §3 already adjudicated 230M-first on deployment and budget grounds; nothing in the brutal review suggests capacity is the binding constraint — data identifiability (#1) is. Swap is one CLI flag later if the confound-controlled eval shows a real margin. **REJECT for v2.**

---

## Top 3 for v2

1. **Same-prompt paired contrastive training (#1)** — the only idea that attacks the root cause (register≡label confound) with data the repo already owns. Everything else is decoration if the model is still doing corpus discrimination.
2. **Eval/calibration integrity package (#2)** — separate calibration/reporting splits, holdout-driven checkpoint selection, min-n gates, bootstrap CIs, same-distribution floor comparison. Makes every other claim in this file testable; without it v2 repeats v1's sin of impressive, meaningless numbers.
3. **Feature fusion + token-loss rebalance + random crops as one training-run bundle (#3+#4+#5)** — all three are small trainer changes that ride the same ≤10h run; together they make the doc head stronger at the FPR tail and the span head actually alive, which is what the product ships.

Sequencing: land #2's harness first (it defines success), then the #3/#4/#5/#10 bundle with #1's pairs, then decide #7–#11 from the resulting holdout numbers.
