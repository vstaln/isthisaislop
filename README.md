# Is This AI Slop?

**ITAIS** for short. GitHub: [`isthisaislop`](https://github.com/vstaln/isthisaislop).

Local checkable-hit detector for AI-style prose, plus a separate `matches_ai_pile` resemblance score. Never an authorship claim.

Python import stays `slopdet`. CLI: `itais` or `slopdet`.

## Colab (this is the training path)

1. Open `notebooks/SlopDetector_Colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **T4 GPU** (CPU also works for the smoke path).
3. Runtime → **Run all**.

Smoke mode finishes on a free T4 without a Hugging Face token. It trains a calibrated `matches_ai_pile` head on weak labels + construction stats, then demos hits on sample text.

Optional: add a Colab secret named `HF_TOKEN` (Hugging Face, with Gemma access) and set `FULL = True` in the config cell to cache Gemma-3-4B residuals. If Gemma is gated or OOMs, the notebook falls back to `Qwen/Qwen2.5-0.5B-Instruct`.

Outputs go to Google Drive `MyDrive/isthisaislop/` when Drive is mounted, otherwise `/content/isthisaislop/`.

## Two lanes

- **Style / construction hits:** named pattern, quoted span, short fix. Checkable.
- **Resemblance:** `matches_ai_pile` vs a human reference slice. Not a percentage of AI-ness. Empty hits render as "Nothing matched."

## License

Apache-2.0 for code. Wikipedia-derived pattern blurbs are isolated in `ontology/patterns.wikipedia.yaml` (CC BY-SA 4.0).
