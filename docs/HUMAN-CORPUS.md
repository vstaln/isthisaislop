# Human corpus — what v2 still needs

Companion to `docs/HANDOFF.md`, written against **v2** (222k docs, provenance-labeled, pair-keyed).
This is a diff, not a from-scratch plan: v2 already solved the parts most detector corpora get wrong.

Fact classes as in the handoff: MEASURED / CITED / ESTIMATE / CONJECTURED / STALE-BY.

## 1. What v2 already gets right

Provenance-based labels rather than model-judged ground truth; `split_hint` pair-keys so a generated
text never lands in a different split from its human source; generator-disjoint and paraphrase
holdouts; a stratified contamination audit across all 27 registers. Those are the expensive
correctness properties and they are done.

## 2. The gap, MEASURED

`uv run python scripts/corpus_probes.py --hf vstalingrady/itais/v2_train_labeled.parquet`
(artifact: `artifacts/corpus_probes.json`, run 2026-08-25 on all 222,067 rows):

| Probe | Result | Threshold | |
|---|---|---|---|
| **register-only** — predict the label from the register tag, never reading a word | **0.989** | ≤ 0.55 | FAIL |
| **length-only** — predict the label from token count alone | **0.705** | ≤ 0.55 | FAIL |

24 of 27 registers carry a single label, covering **180,104 docs — 81% of the corpus**. Median length
is 395 words for human rows and 191 for AI rows. Only three registers contain both labels at all:

| Mixed register | Docs | AI fraction |
|---|---|---|
| coai | 38,784 | 0.774 |
| raid_abstracts | 1,581 | 0.908 |
| raid_books | 1,598 | 0.909 |

That is 41,963 docs, 19% of the corpus, and two of the three are 91% AI. Inside those mixed registers
the length shortcut mostly disappears (length-only 0.56, medians 142 vs 131 words), which confirms the
confound lives *between* registers rather than inside them.

**What this means:** a model trained on v2 can reach ~0.99 AUROC by identifying the register and never
reading the writing. Any headline number measured on v2 as it stands is uninterpretable — not wrong,
just not evidence about detection. Pair-keys stop a *pair* from straddling splits; they don't stop a
*register* from being diagnostic on its own.

### Fixes, cheapest first

- **m4's human half was dropped by a label-blind cap.** `scripts/fetch_v2.py:fetch_m4` keeps both
  labels (`generator="human" if lab == 0`), but the `per_domain: int = 8000` cap counts by domain
  only, and each of the five m4 domains landed at ~7,7xx rows that are **100% AI** — the cap filled
  from a label-ordered source file before any human row was reached. Capping per (domain, label)
  recovers roughly 38k human docs across news/wiki/QA/how-to registers and converts five single-label
  registers into mixed ones. This is the single highest-value change in this document.
- **Paired data split across two register names re-creates the shortcut.** `wiki_intro` (14,543 human)
  and `wiki_intro_gpt` (13,629 AI) are the same register under two tags, as are the `hc3_*` /
  `hc3_*_gpt` families. Merge them into one register with two labels: per-register AUROC is currently
  undefined on both halves, so those slices report nothing today.
- **WritingPrompts is 28% of the corpus and human-only.** The `rewrite_pair` / `respond_pair`
  generators that produced the 12,803 contrastive rows already do exactly the needed job; run them
  over WP prompts.
- **Match lengths.** 395 versus 191 median words is worth 0.705 AUROC on its own. Window or truncate
  to a matched distribution per register before training.
- **The AI pile is mostly 2023-era.** Top generators are gpt-3.5 (13,629), chatGPT (11,455), davinci
  (10,982), cohere (8,641), dolly (7,916); the current-gen rows (gemma-4-26b 6,293, laguna-s-2.1
  2,913, ox-alpha-free 2,465) are a small minority. The plan doc's own premise is that 2023-era slop
  is stale, so the mixture should reflect that.

## 3. The gap that matters for the product: the 2×2

| | not sloppy | sloppy |
|---|---|---|
| **human** | edited journalism, literary fiction, good technical writing | **missing**: marketing copy, SEO how-tos, press releases, corporate blogs, low-effort reviews |
| **AI** | **thin**: anti-slop-prompted generations, human-edited AI drafts | default chat-model output (well covered) |

The product's premise is that slop and machine-authorship are different axes — `why_slop` names style
hits, `matches_ai_pile` is a separate resemblance line. If the corpus only ever pairs clean-human with
sloppy-AI, the two lanes are trained on the same axis and the separation is decorative. Filling the
other two cells is what makes the design real:

- **Human-sloppy.** Densest supply is C4's 2019 crawl (ODC-BY, pre-LLM by construction): "about us"
  pages, landing pages, listicles, press releases. Filter by URL and template shape rather than by
  hand. This is the hard negative — text that should light up `why_slop` while `matches_ai_pile`
  stays human.
- **AI-clean.** Falls out of an asset already in this repo: prompt current models with the 415
  ontology patterns as a *negative* instruction ("do not use these phrases or constructions"), plus a
  human-edit pass over AI drafts. These are the hard positives a naive detector misses. Costs API
  calls, not GPU time, and reuses the v2 generation pipeline unchanged.

## 4. Era: the confound v2 cannot see yet

Every human source in v2 predates the assistant era (WritingPrompts, Gutenberg, 2004 blogs, arxiv
abstracts, HC3's 2022 crawl). All AI text is 2026. So era correlates almost perfectly with the label,
and a model can score well by dating the text — then fail on the only writing the app will ever see,
which is 2026 human writing by people who now read a lot of LLM prose.

Fix without retraining anything: tag every row `era ∈ {pre, post}`, keep training on `pre`, and add a
**post-2022 verified-human** slice used only to measure false positives. Sources with trustworthy
recent timestamps: Stack Exchange dumps (CC BY-SA 4.0, per-post dates), Wikipedia revisions after
2023, and hand-collected recent writing from known humans. A rising FPR on `post` is the model reading
the calendar, and today nothing in the eval would reveal it.

Related fairness slice, eval-only: **ICNALE** (5,600 learner essays, ~1.3M words, free after
registration, CITED 2026-08-18). Detectors systematically over-flag non-native English writers; without
this slice we would not know whether ours does.

## 5. Acceptance probes — run these on v2 as it stands

Deliberately dumb models. Each one that succeeds is a shortcut the real model is also free to take.

| Probe | What it is | Pass condition |
|---|---|---|
| Length-only | logistic regression on token count alone | AUROC ≤ 0.55 — **MEASURED 0.705 on v2** |
| Register-only | predict the label from the source/register tag alone | AUROC ≤ 0.55 — **MEASURED 0.989 on v2** |
| Topic-only | TF-IDF over content words with style words stripped | AUROC ≤ 0.65 |
| Format-only | punctuation, markdown markers, whitespace, casing | AUROC ≤ 0.60 |
| Era probe | classifier trained on human text to predict `era`, applied to the AI pile | must not separate AI from `pre`-human better than chance |

Thresholds are ESTIMATE — judgement calls to tighten once a build passes them. `scripts/corpus_probes.py`
runs the first two and exits nonzero on failure, so it can gate a corpus build; topic-only, format-only
and the era probe are still to be written. Record every result in `artifacts/corpus_probes.json`.

## 6. Licensing tiers

Keep two pools in the manifest and never mix them:

- **Trainable** — permissive or public-domain human text (C4 ODC-BY, Stack Exchange and Wikipedia
  CC BY-SA, Gutenberg public domain) plus our own generations. Only these may touch shipped weights.
- **Eval-only** — NC-licensed, registration-gated or ambiguous. PERSUADE belongs here until its
  licence conflict resolves: the corpus repos state CC BY-NC-SA 4.0 while the publisher's dataset
  pages state CC BY 4.0 (CITED, both checked 2026-08-18). ICNALE, Yelp and anything with unclear terms
  sit here too.

Measuring on eval-only data is fine; training on it puts terms we can't honour into the artifact users
download, which is the one thing "free forever" cannot survive.

## 7. Order of work

1. Fix the m4 cap to be per (domain, label) and re-merge (§2) — ~38k human docs, five registers become mixed.
2. Merge `*_gpt` register names into their human twins (§2), then re-run `scripts/corpus_probes.py`.
3. Pair the WritingPrompts register with the existing generators, and match lengths (§2).
4. Pull human-slop from C4-2019 and generate the AI-clean cell (§3).
5. Add the `post`-era FPR slice and ICNALE as eval-only (§4).
