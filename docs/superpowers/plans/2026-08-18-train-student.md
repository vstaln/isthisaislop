# Implementation Plan: ITAIS v1 — a ZeroGPT-class scorer you can run anywhere

> **Date:** 2026-08-18
> **Budget:** $0 — Colab free T4, 15 GB Google Drive.
> **Replaces** the earlier train-student plan (Gemma teacher → 40M student → 5 heads). That pipeline is deferred (see "Deferred" below).

## Goal

Select text → "Is this AI slop?" → named why-slop, named why-human, per-sentence lean, plus a calibrated `matches_ai_pile` resemblance line. Never an authorship claim.

- Product shape: mobile-first, later (ONNX int8 / transformers.js — no UI in v1).
- Copy rule stays: never "87% AI", never "written by ChatGPT". The number is a calibrated pile-similarity, compared against a named human reference class.
- **Why** is checkable hits (quoted span + pattern id + fix) and human cues (weekday, number, contraction). Not a generated essay.

coai is arxiv abstracts vs LLM paraphrases (claude-haiku-4.5, gemini-3-flash, gpt-oss-120b, gpt-5-nano). That trains academic-paraphrase resemblance. Ordinary blog/email slop is named by the ontology. Both lanes ship.

## Architecture (v1)

```
coai/ai-text-detection-training (Dec 2025, 62,460 train + 11,022 test, balanced)
  human: arxiv abstracts + paraphrases
  ai:    claude-haiku-4.5, gemini-3-flash-preview, gpt-oss-120b, gpt-5-nano
        ──► fine-tune FacebookAI/roberta-base (125M, sequence classifier, 1–2 epochs, T4 ~15–30 min)
        ──► eval on coai test: AUC, accuracy
        ──► calibrate threshold at 1% FPR on the human slice of test
        ──► export artifacts/roberta/ (safetensors + tokenizer + calibration.json + manifest.json)
```

Corpus is fetched as parquet files straight off the Hub (no dataset scripts — the HC3 lesson). Data is **current-gen model text**, which is the whole point: 2023-era slop lists are stale.

## Why no teacher / no student / no Bonsai

- The teacher+student detour existed to make a *phone-sized* model from a 4B teacher. A 125M RoBERTa **is** phone-sized when trained directly: int8 ≈ 125 MB, int4 ≈ 60 MB. No distillation needed.
- 1-bit models (Bonsai) can't be fine-tuned with existing toolchains. Rejected.
- The LR-on-features smoke path stays in `colab_pipeline.py` only as a CPU fallback when transformers/torch can't install.

## Steps

1. `colab_pipeline.py`: `fine_tune_roberta(root, epochs, max_len)` — download coai train+test parquet (cached in `data/`), fine-tune roberta-base, eval AUC, calibrate at 1% FPR on human test slice, export bundle.
2. `notebooks/build_colab_nb.py`: notebook = config → write package → train → demo. No Gemma cell, no distill cell.
3. After training: push `artifacts/roberta/` to a public HF repo (`vstaln/isthisaislop-roberta` or similar) so the future mobile/web packaging can load it without Colab.
4. Eval targets on coai test: AUC ≥ 0.95, accuracy ≥ 0.88, FPR ≤ 1% at operating threshold. Manual eyeball of 50 test preds.

## Success criteria

- [ ] `fine_tune_roberta` runs end-to-end on a free T4, one session, < 40 min.
- [ ] AUC ≥ 0.95 on coai test; FPR ≤ 1% on the human slice at operating threshold.
- [ ] `artifacts/roberta/` contains model + tokenizer + calibration.json + manifest.json (trained_on, never_trained_on, auc, threshold).
- [ ] Demo scores two samples and prints a verdict line (never a percentage of AI).
- [ ] Model loads from plain transformers `AutoModelForSequenceClassification.from_pretrained` (portable to ONNX/transformers.js later).

## Shipped locally (no GPU)

- `slopdet.explain`: why_slop / why_human / per-sentence lean.
- CPU `matches_ai_pile` logistic head on coai 8k (`scripts/train_cpu_scorer.py`).
- Span stitch + `pure_docs` calibration (fixes empty human_scores on mixed docs).

## Deferred (explicitly not v1)

- Teacher residual distillation, J-lens, 40M student, Bonsai/ternary — revisit only if the 125M scorer measurably plateaus.
- Browser extension, toast UI, mobile app, Twitter posting — after the model exists and scores well.
- RAID / HC3 / FineWeb piles — coai alone is sufficient; add sources only if OOD eval shows a gap.