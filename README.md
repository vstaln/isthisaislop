# Is This AI Slop?

**ITAIS** for short. GitHub: [`isthisaislop`](https://github.com/vstaln/isthisaislop).

Local detector you can run on selected text: named slop patterns, named human-style cues, per-sentence lean, plus a separate `matches_ai_pile` resemblance score. Never an authorship claim.

Python import stays `slopdet`. CLI: `itais` or `slopdet`.

```bash
uv run itais "Here's the thing, we leverage robust pipelines."
uv run itais --json "Thursday mornings at the clinic were empty."
```

`why_slop` quotes the span and names the pattern. `why_human` quotes anchors (a weekday, a number, a contraction). `lean` is `slop`, `human`, `mixed`, or `unclear` from the sentences — not "written by ChatGPT."

## Colab (neural span model)

1. Open `notebooks/SlopDetector_Colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**.

Trains `roberta-base` as a token classifier on stitched coai sentences (arxiv abstracts vs current-gen LLM paraphrases). Exports `artifacts/roberta-span/`. About 20–40 min on a free T4.

coai is **academic paraphrase**, not blog/email slop. The regex ontology is what names "here's the thing" / "leverage" / em-dashes on ordinary prose. Use both.

CPU scorer (no GPU):

```bash
uv run python scripts/train_cpu_scorer.py
```

## Two lanes

- **Style / construction hits:** named pattern, quoted span, short fix. Checkable.
- **Resemblance:** `matches_ai_pile` vs a human reference slice. Not a percentage of AI-ness. Empty hits render as "Nothing matched."

## License

MIT for code. Wikipedia-derived pattern blurbs are isolated in `ontology/patterns.wikipedia.yaml` (CC BY-SA 4.0). Anti-slop phrases in `ontology/patterns.slop.yaml` are Apache-2.0 (sam-paech/antislop-sampler).
