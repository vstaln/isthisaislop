# Human corpus spec

Companion to `docs/HANDOFF.md`. What counts as human text here, where it comes from, and the probes a
build must pass before anything trains on it.

Fact classes as in the handoff: MEASURED / CITED / ESTIMATE / CONJECTURED / STALE-BY.

## 1. The problem with the human pile we have

Today's human side is arxiv abstracts (coai), 2004 blogs, pre-2022 Reddit fiction, and public-domain
Gutenberg. Every one of those is human beyond doubt, and together they still make a corpus that
teaches the wrong thing, because each differs from the AI side along an axis that has nothing to do
with who wrote it:

- **Era.** Human text from 2004–2021 versus AI text from 2026 means topics, entities, slang, and
  formatting conventions separate the classes perfectly. The model can hit high AUROC by dating the
  text. It will then fail on 2026 human writing, which is the only kind the app will ever see.
- **Register.** Fiction and academic abstracts on the human side versus chat-model prose on the AI
  side teaches register, not provenance.
- **Slop.** All-human text that is edited and literary, all-AI text that is generic, means the
  resemblance lane and the style lane collapse into the same axis — and the product's entire premise
  is that they are different. A human press release must score human-and-sloppy, or the two lanes are
  a costume for one classifier.

## 2. What the corpus has to span: the 2×2

Both lanes need all four cells populated, not just the diagonal.

| | not sloppy | sloppy |
|---|---|---|
| **human** | edited journalism, literary fiction, good technical writing | marketing copy, SEO how-tos, press releases, corporate blogs, low-effort reviews |
| **AI** | anti-slop-prompted generations, human-edited AI drafts | default chat-model output (what we already have) |

Two cells are cheap and we don't have them:

- **Human-sloppy** is the hard negative that decides whether the product is honest. Sources in §3;
  C4-era corporate web pages are the densest supply.
- **AI-clean** comes free from the asset already in this repo: prompt current models with the 415
  ontology patterns as a *negative* instruction ("do not use these phrases or constructions"), plus
  the human-edited-AI case. These are the hard positives, and they are the texts a naive detector
  misses. Generating them costs API calls, not GPU time.

## 3. Sources

Dates are what makes a human label trustworthy: anything written after 2022-11-30 may be
LLM-assisted, so it is tagged `era=post` and used for false-positive measurement rather than as clean
human training data. Licences below are CITED from each source, checked 2026-08-18, and must be
re-checked before shipping weights — see §6.

| Register | Source | Human-clean? | Licence / access | Notes |
|---|---|---|---|---|
| Encyclopedic | Wikipedia dump, revisions before 2022-11 | yes, revision-dated | CC BY-SA | filter by revision timestamp, not article age |
| Q&A / technical | Stack Exchange data dump | yes, post-dated | CC BY-SA 4.0 | huge, per-post timestamps, informal-technical register |
| Corporate / SEO / marketing | C4 (Common Crawl, April 2019) | yes, pre-LLM crawl | ODC-BY | **the human-slop supply**; filter for "about us", landing pages, listicles |
| News | CC-News pre-2019 | yes | per-publisher | edited professional prose |
| Personal blog | Blog Authorship Corpus (2004) | yes | research use | already on disk; era-skewed, cap it |
| Forum fiction | WritingPrompts | yes, pre-2022 | Reddit user content | already on disk |
| Modern fiction | SCP wiki | mostly | CC BY-SA | **needs a pre-2023 cutoff**; the wiki is live and AI-assisted entries exist |
| Literary fiction | Project Gutenberg | yes | public domain | already on disk; pre-1929 English, cap it hard |
| Email | Enron email corpus | yes (2001–02) | public | the only large real email register available |
| Reviews | IMDb (Maas 2011), Amazon Reviews 2018 | yes | research use | low-effort human writing, close to slop |
| Student essays | PERSUADE 2.0 / ASAP 2.0 | yes, pre-2022 | **conflict**: the corpus repos state CC BY-NC-SA 4.0, the publisher's dataset pages state CC BY 4.0 | resolve before training; eval-only until then |
| Non-native English | ICNALE (5,600 essays, ~1.3M words) | yes | free after registration, research use | the fairness slice — detectors falsely flag L2 writers |
| Transcribed speech | podcast/parliamentary transcripts | yes | varies | useful: human text with no editing pass at all |

`data/raw/manifest.json` already exists — every row added must land there with source, licence, date
range, row count and sha256.

## 4. Construction rules

- **Topic pairing.** For each human seed, generate the AI counterpart on the *same title and length
  target*, the way coai pairs an arxiv abstract with its paraphrase. Unpaired piles let topic leak
  into the label.
- **Length matching.** Match the token-length distribution per register between classes, then window
  both to the training length. A length-only classifier must be near chance (§5).
- **Formatting normalization.** AI text arrives with markdown; scraped human text arrives as
  HTML-to-text. Normalize both (or randomly re-apply markdown to human text), or the model learns to
  detect `**bold**` and em-dash density instead of writing.
- **Era stratification.** Every doc carries `era ∈ {pre, post}`. Train on `pre` plus paired AI;
  measure FPR on `post` human writing separately and report it. A rising FPR on `post` is the model
  learning the calendar.
- **Dedup.** MinHash across the whole corpus and across splits; split by source document and register
  so nothing near-duplicate straddles train and test.

## 5. Acceptance probes — a corpus is broken until these pass

Cheap, deliberately dumb models. Each one that succeeds is a shortcut the real model would take too.

| Probe | What it is | Pass condition |
|---|---|---|
| Length-only | logistic regression on token count alone | AUROC ≤ 0.55 |
| Topic-only | bag-of-nouns / TF-IDF over content words with style words stripped | AUROC ≤ 0.65 |
| Era probe | classifier trained to predict `era` from human text only, applied to the AI pile | must not separate AI from `pre`-human better than chance |
| Format-only | punctuation, markdown markers, whitespace, casing features | AUROC ≤ 0.60 |

ESTIMATE on the thresholds: they are judgement calls, not derived — tighten them once the first real
build gives a distribution to look at. Record every probe result in `artifacts/MANIFEST.json`.

## 6. Licensing and the "free forever" goal

The shipped weights must be redistributable. Two tiers, kept apart in the manifest:

- **Trainable**: permissive or public-domain human text, plus our own generations. This is what the
  shipped model is allowed to see.
- **Eval-only**: anything NC-licensed, registration-gated or ambiguous (PERSUADE until the licence
  conflict resolves, ICNALE, Yelp, anything with unclear terms). Measuring on it is fine; training on
  it contaminates the artifact we hand to users.

Weights are not a derivative of the data in the same way a redistribution is, but the manifest should
still record exactly what went in, so the question can be answered rather than argued.

## 7. Build order

Cheapest first, and each step ends with the probes in §5 re-run:

1. **Human-slop from C4-2019** — the missing cell that matters most, one crawl, permissive licence.
2. **Stack Exchange + Wikipedia pre-2022** — bulk modern-ish human prose with hard timestamps.
3. **AI-clean generations** — ontology-as-negative-prompt over current models, paired to the seeds
   from 1 and 2.
4. **Reviews + email + transcripts** — the unedited-human registers.
5. **Eval-only slices** — ICNALE, PERSUADE, `post`-era human web text, for the fairness and
   era-drift reports.

ESTIMATE of scale: ~30k docs per register across ~8 registers ≈ 240k docs ≈ 100M tokens at 512
tokens, which is ~2–3 h/epoch for the 230M encoder on a T4 — the same order as the current plan, so
the corpus can grow this far without changing the training story.
