# Implementation Plan: Train a local slop-detector student (Gemma-3-4B teacher → 40M student → 4 heads + contrast)

> Saved from the [planner](80b9b523-705f-4f9d-a435-68c99b96cc8f) (read-only; could not write itself).
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-08-18
**Budget:** $0 — Colab free T4, Kaggle T4/P100 backup, 15 GB Google Drive.

**Kobak citation (resolved):** the paper is [arXiv:2406.07016](https://arxiv.org/abs/2406.07016) (also in `noslop/research/REFERENCES.md`). `noslop/README.md` citing `2407.07004` is a typo. Ontology source URLs use **2406.07016**.

## Goal

Ship a locally-runnable trained model that, given a piece of general English web text, returns **checkable hits** (named pattern + quoted span + short fix) and, separately, a **resemblance score against an AI pile** — never an authorship claim. v1 is the model plus the hit schema. The browser overlay is packaging and is out of scope except for a frozen interface stub.

The two lanes never merge in the model, the schema, or the copy:

| Lane | What it asserts | Highlight unit | Evidence type |
|---|---|---|---|
| 1. Style / construction hits | "this named pattern appears here" | style = smallest span; construction = paragraph or whole piece | checkable by the reader |
| 2. Resemblance | `matches_ai_pile` only | sentence | statistical, not checkable |

Lane 2 is never rendered as a percentage of AI-ness and never as a verdict.

## Architecture

```
 ┌─ Phase 0 ────────────────────────────┐
 │ ontology/patterns.*.yaml (frozen ids) │
 └──────────────┬───────────────────────┘
                │ weak labels
Phase 1 RAID-train ─┐                  ▼
        HC3         ├─► subsample ─► weaklabel.py ─► construction.py ─► corpus.jsonl (~100k)
        FineWeb-Edu │
        T4-generated┘                  │
                                       ▼
Phase 2 Gemma-3-4B-it 4-bit (teacher, L17) ──► token residual cache (.npy shards, Drive)
        jacobian-lens ──► J-space verbalizer ──► FILTER ──► closed 512-word list ──► jspace[]
                                       │
                                       ▼ MSE, z-normed, token-level
        student: 128-d embed, 4 layers, hidden 256, 4 attn heads, 2560-d out (~40M)
                                       │
Phase 3                                ▼ freeze trunk
        head_lexical  head_rhetorical  head_construction  head_jspace  head_contrast
                                       │
                                       ▼
        verify.py (SHA-pinned, fail-closed) ─► report.py ─► hit schema JSON ─► extension/INTERFACE.md (stub)
```

The teacher never ships. Binoculars-style two-model scoring is a diagnostic only (two 7B models will not run locally).

## Tech Stack

- Python 3.11, PyTorch 2.x, `transformers`, `bitsandbytes` (4-bit NF4), `accelerate`, `datasets`, `safetensors`, `pyyaml`, `regex`, `pyarrow`, `numpy`, `scikit-learn` (calibration only).
- Teacher: `google/gemma-3-4b-it` loaded 4-bit, or `unsloth/gemma-3-4b-it-bnb-4bit` if the plain load OOMs on a T4. [inferred]
- J-lens: `anthropics/jacobian-lens`, Apache-2.0.
- Code license Apache-2.0. CC BY-SA content quarantined (see Phase 0).
- No paid APIs anywhere in the pipeline. No LLM-as-judge for labels, per AgriciDaniel's finding that judge agreement sits near chance.

## Global Constraints

1. **Storage is the binding constraint, not compute.** Free Drive is 15 GB. Full token-level residual caching of 100k docs × 512 tokens × 2560-d fp16 is ≈262 GB. It does not fit and never will. The cache is therefore *position-subsampled*, not doc-subsampled: see Phase 2 budget.
2. **Every long-running cell is resumable.** Shard + manifest. A Colab disconnect costs at most one shard.
3. **RAID-test has no public labels.** Never train on it, never eval on it as if labeled.
4. **No LLM-as-judge anywhere in label generation.**
5. **No pattern lands in the ontology without an id, a span type, and a fix blurb.** A pattern with no fix is not evidence, it's a vibe.
6. **Fail closed.** Any artifact hash mismatch returns an empty hit list with `status: "unverified_artifact"`. It does not silently fall back to regex-only.

## Assumptions

**[verified]** — read directly from files during planning:

- `no-ai-slop/SKILL.md` defines the detect schema as *named pattern + quoted line + few-word fix*, explicitly forbids scoring the draft and forbids guessing whether AI wrote it. This is the schema v1 implements.
- `no-ai-slop/SKILL.md` supplies concrete named patterns usable as ontology ids: binary contrasts, throat-clearing openers, faux-insight setups, colon reveals, superficial analysis (trailing `-ing`), importance puffery, interpretive metadiscourse, weasel attribution, fake-strong verbs, synonym cycling, negative listing, dramatic fragmentation, robotic rhythm, rhetorical setups, fake-profound kickers, summary-recap endings, formatting slop, em dashes. Plus a hard ban list (~30 words) and two empty-phrase lists.
- `anti-ai-slop-writing/SKILL.md` supplies structural ids: rule-of-three, uniform sentence length, parataxis, hedging seesaw, corporate pep talk, identical paragraph structure, bullet overuse, "As [role], I..." openers, cross-section parallelism, passive construction, mandatory paragraph transitions; plus punctuation budgets (em dash ≤1/500 words, exclamation ≤1/1000, ellipsis ≤1/piece).
- `noslop/README.md` states noslop is a craft tool, not a detector, and owns the evenness/construction framing. It is not the product. New tree confirmed.
- Kobak is arXiv **2406.07016**, not 2407.07004.

**[inferred]** — from the brief, not independently checked; each has a stated fallback:

- RAID train CSV is at `https://dataset.raid-bench.xyz/train_none.csv`; HF mirror `liamdugan/raid`. Fallback: HF mirror only.
- Fast's recipe (100k RAID, MSE to Gemma-3-4B L17 z-normed, 4-layer/256-hidden/4-head/128-embed/2560-out, contrast 0.93 AUC vs Gemma 0.95, raid-finetune overfits) is accurate as reported. Fallback: the contrast default stands regardless; AUC targets are goals, not claims.
- ~40M params comes almost entirely from the embedding table (Gemma tokenizer ≈262k vocab × 128 = ~33.5M). If a smaller tokenizer is substituted the param count drops and that is fine.
- Gemma-3-4B hidden width is 2560, so L17 residual is 2560-d. Verify with one `print(model.config)` in Phase 2 cell 1 before caching anything.
- A Pangram evaluation slice is obtainable. Fallback if not: OOD eval = RAID adversarial-attack subset + fresh T4-generated text from a model family absent from training.
- StoryScope parquet contains AI story text with 30-core labels but no human story text (Books3 gap). Consequence: it can only supply *AI-side* construction supervision, so it is **deferred out of v1** and cannot block anything.
- `DrRiceIO7/SlopReview` and Slopasaurus exist as small fiction-trope sets. Optional garnish, never load-bearing.
- J-lens runs on a 4-bit model. If it requires fp16 weights, fall back to fitting the lens on a smaller model of the same family or drop to the mean-pool rollback.

---

## Phase 0 — Ontology freeze (Day 1, CPU only, no GPU)

Nothing downstream is meaningful until the label vocabulary stops moving. Freeze it first, version it, never renumber ids.

- [ ] Create the tree:

```
/home/vstaln/slop-detector/
  README.md  LICENSE  THIRD_PARTY_NOTICES.md  pyproject.toml  .gitignore
  ontology/
    schema.json
    patterns.core.yaml          # from no-ai-slop + noslop bans + anti-ai-slop-writing
    patterns.wikipedia.yaml     # CC BY-SA 4.0 ONLY — quarantined
    patterns.rhetorical.yaml    # Reinhart/Biber-derived grammatical heuristics
    LICENSES.md
  src/slopdet/
    ontology.py  weaklabel.py  construction.py  teacher.py  jlens.py
    student.py  heads.py  calibrate.py  verify.py  report.py  cli.py
  notebooks/
    01_build_dataset.ipynb  02_teacher_cache_jlens.ipynb
    03_distill_student.ipynb  04_heads_and_eval.ipynb
  data/                         # gitignored
  artifacts/                    # MANIFEST.json + weights, gitignored except manifest
  eval/slices/  eval/reports/
  extension/INTERFACE.md        # stub only
  tests/
```

- [ ] Write `ontology/schema.json`. Every entry requires exactly these keys:

```yaml
- id: colon_reveal              # stable, snake_case, never reused
  lane: style                   # style | rhetorical | construction
  unit: span                    # span | sentence | paragraph | piece
  detector: regex               # regex | heuristic | model_only
  pattern: '(?m)^[A-Z][^.!?\n]{2,60}: [a-z]'
  fix: "Rewrite as a plain sentence."
  source: no-ai-slop            # provenance
  license: MIT-compatible       # or CC-BY-SA-4.0
  min_len_words: 0              # gate: some patterns only apply to long text
  paper: null                   # e.g. reinhart-2410.16107
```

- [ ] Populate `patterns.core.yaml` from the two skills. Target ≈70–90 ids. Concretely: 18 named patterns + ~30 banned words (prefer one id per *class*, e.g. `ban_delve_class`, so hit counts stay interpretable) + 11 empty adverbs + ~18 empty phrases + 11 structural rules from anti-ai-slop-writing + 3 punctuation budgets.
- [ ] Populate `patterns.wikipedia.yaml` from the deslop rendering of Wikipedia's *Signs of AI writing*. **Licensing move:** CC BY-SA 4.0 is share-alike and viral on derivative *text*. Keep every descriptive sentence and fix blurb derived from it in this file alone, mark the file's header with attribution and the CC BY-SA 4.0 link, list it in `THIRD_PARTY_NOTICES.md`, and keep `src/` and the other YAMLs free of its prose. Regex strings are functional and likely uncopyrightable, but the descriptions are not — do not paste them into Apache-2.0 source. Not legal advice; if in doubt, ship the file separately and load it as data.
- [ ] Populate `patterns.rhetorical.yaml` from Reinhart 2410.16107 / PNAS 2025 Biber dimensions: present-participial clauses (`, VBG`), nominalization density (`-tion|-ment|-ness|-ity` per 1k words), copula avoidance (ratio of `be`-verbs to lexical verbs vs human baseline), that-complement rate, attributive adjective stacking. These are `detector: heuristic` and need a POS tagger — use `spacy` `en_core_web_sm` on CPU during dataset build only, never at inference.
- [ ] Implement `ontology.py`: loads all three YAMLs, rejects duplicate ids, compiles regexes once, emits `ONTOLOGY_SHA256` over the concatenated canonical YAML bytes.
- [ ] `tests/test_ontology.py`: schema validation, no duplicate ids, every regex compiles, every entry has a non-empty `fix`, licenses declared.
- [ ] Tag `ontology-v1`. Ids are append-only after this point.

**Rollback:** ontology is pure data with no dependencies. If a pattern proves unlabelable, set `enabled: false` rather than deleting the id — deleting shifts head output dimensions and invalidates trained heads.

---

## Phase 1 — Corpus assembly (Day 2, mostly CPU, one short T4 session)

Notebook `01_build_dataset.ipynb`. Output: `data/corpus.jsonl` (~100k docs) plus `data/splits.json`.

Record schema:

```json
{"id": "...", "text": "...", "source": "raid|hc3|fineweb-edu|selfgen",
 "model": "human|gpt4|mistral|...", "style_hits": [], "construction": {},
 "jspace": [], "pile": 0}
```

`style_hits[]` entries: `{"id": "...", "start": 0, "end": 0, "unit": "span"}`. `jspace[]` is filled in Phase 2, empty here. `pile` is 1 for machine-authored, 0 for human — this is the contrast target and is a *pile membership* label, not a truth claim about any user's future text.

- [ ] **RAID train.** `wget -c https://dataset.raid-bench.xyz/train_none.csv -O /content/raid_train.csv`. If that 404s: `datasets.load_dataset("liamdugan/raid", "raid")` and take the train split. Subsample to ~50k rows stratified by (domain, model, decoding, attack). Keep `attack == none` for the main train pool; reserve attacked rows for the OOD eval slice. **RAID-test is never touched.**
- [ ] **HC3.** `load_dataset("Hello-SimpleAI/HC3", "all")`. Explode into paired human answers and ChatGPT answers, ~20k docs, balanced.
- [ ] **FineWeb-Edu human slice.** `load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)`, take 20k docs of 200–2000 words. Filter by `dump` to pre-2022 crawls where available to reduce contamination by post-ChatGPT web text.
- [ ] **Self-generated web-like text (T4, ~1 hour).** Generate 10k short docs across article / email / LinkedIn post / marketing copy / support answer, from 2–3 open models that are *not* the teacher, e.g. `Qwen/Qwen2.5-1.5B-Instruct` and `microsoft/Phi-3-mini-4k-instruct` in 4-bit. Vary temperature 0.6–1.1 and top-p 0.85–0.98 so decoding is not a giveaway. Mark `source: selfgen`.
- [ ] **Weak-label spans.** `weaklabel.py` runs the compiled ontology over every doc, writing character offsets. Overlapping hits from different ids all survive. Hits whose `min_len_words` gate fails are dropped.
- [ ] **Construction stats.** `construction.py`, cheap and deterministic:
  - `burstiness` = std(sentence_len)/mean(sentence_len); plus `adjacent_contrast` = count of adjacent sentence pairs with |Δlen| ≥ 20.
  - `evenness` = mean pairwise cosine similarity between per-paragraph feature vectors [mean sentence length, type-token ratio, comma density, subordinator density, mean word length]. High = machine-like.
  - `recap_closure` = content-word overlap between the final paragraph and the union of earlier paragraph-initial sentences, plus a closure-lexicon hit ("In conclusion", "Ultimately", "Overall").
  - `over_explain` = interpretive-metadiscourse hits + trailing-participial (`highlighting|underscoring|reflecting|showcasing`) rate, per 1k words.
  - `portability` = fraction of sentences containing zero proper nouns, digits, or dates.
- [ ] **Splits.** `train` / `dev` / `test` disjoint by document. Held-out slices kept entirely out of train: (a) 5k RAID rows, stratified, (b) all RAID attacked rows, (c) 3k FineWeb-Edu, (d) Pangram slice if obtainable.
- [ ] `data/corpus.jsonl` gzipped to Drive with a `corpus_sha256`.
- [ ] `tests/test_weaklabel.py`: hand-write 20 sentences with known patterns, assert exact span recovery.

**Rollback:** if any single source fails to download, the corpus builds without it — the notebook takes a `SOURCES` list and each loader is independently skippable. The floor is RAID + FineWeb-Edu, which alone is sufficient for the contrast head.

---

## Phase 2 — Teacher cache, J-lens, student distillation (Days 3–4, T4)

**Cache budget.** Token-level residuals at 2560-d fp16 cost 5,120 bytes per token position. Cache **20,000 documents × 64 stratified token positions each**:

```
20,000 × 64 × 2560 × 2 bytes = 6.55 GB
```

That fits free Drive with headroom, keeps full 2560-d targets, and still gives **1.28M token-level targets**. Positions are sampled stratified across the sequence (not the first 64). The other ~80k corpus docs carry weak labels and construction stats only; they train the heads and the contrast head, which do not need teacher residuals.

If Drive is tighter than expected, the ordered degradation is: 20k → 12k docs, then 64 → 48 positions, then fp16 → a fixed seeded 1024-d Johnson-Lindenstrauss projection (student out becomes 1024-d; note it in the manifest).

Notebook `02_teacher_cache_jlens.ipynb`:

- [ ] Cell 1, before anything expensive: load `google/gemma-3-4b-it` 4-bit, `print(model.config.hidden_size, model.config.num_hidden_layers)`. Confirm 2560 and that L17 exists. If OOM, switch to `unsloth/gemma-3-4b-it-bnb-4bit`. If hidden size differs, update the student output dim and record it.
- [ ] Implement `teacher.py`: forward with `output_hidden_states=True`, take `hidden_states[17]`, chunk text to 256 tokens, batch 4, `torch.no_grad()`, fp16. Sample 64 positions per doc with a per-doc seeded RNG so the sample is reproducible.
- [ ] **Sharded, resumable write.** 40 shards × 500 docs. Each shard writes `resid_{i:03d}.npy` (shape `[500, 64, 2560]`, fp16, ~164 MB) plus `index_{i:03d}.parquet` (doc_id, token positions, token ids), then appends the shard id to `artifacts/cache_manifest.json`. On restart the notebook skips shards already in the manifest. Expected throughput on a free T4 at 4-bit, seq 256, batch 4: roughly 3–6 docs/sec, so 20k docs is **60–110 minutes**. Budget two sessions.
- [ ] Compute z-normalization statistics (per-dimension mean and std) over the first 5 shards only, freeze them into `artifacts/znorm.npz`, and apply them at training time. Do not recompute per epoch.
- [ ] **J-lens fit.** Install `anthropics/jacobian-lens`. Fit on 500 prompts drawn from the corpus (250 human, 250 machine, spread across sources). Output: a verbalizer that maps a residual to in-context words.
- [ ] **J-space filtering — the 0xsero e164 lesson.** Apply all four filters and log how many candidates each one kills:
  1. Drop pure digits, punctuation, whitespace, and sub-3-character fragments.
  2. Drop any token appearing verbatim in the source document (surface echo, not a latent property).
  3. Drop any token whose activation correlates |r| > 0.6 with document length, digit count, or non-ASCII count across the calibration set.
  4. Intersect what survives with a **frozen 512-word ontology-relevant allowlist** built from the Phase 0 pattern vocabulary.

  The result is `artifacts/jspace_vocab.json`, a **closed** 512-word list.
- [ ] Write `jspace[]` back into the corpus records for the 20k cached docs as sparse `(word_index, weight)` pairs.

Notebook `03_distill_student.ipynb`:

- [ ] `student.py`: embedding 262k × 128, 4 transformer layers, hidden 256, 4 attention heads, output projection 256 → 2560. Gemma tokenizer, so student and teacher token indices align exactly.
- [ ] Train: MSE against z-normed L17 residuals at the 64 cached positions per doc. AdamW, lr 3e-4, cosine schedule, warmup 500 steps, batch ~4096 token targets, 20 epochs over 1.28M targets. Checkpoint every 15 minutes to Drive; resume on reconnect. Expected 2–4 hours on a T4.
- [ ] Log dev-set MSE and cosine similarity to teacher per epoch. Stop early if dev cosine plateaus for 3 epochs.
- [ ] Export `artifacts/student.safetensors` plus `artifacts/student_config.json`.

**Rollback:** if the token-level cache cannot finish on a T4, fall back to the **mean-pool Fast recipe**: one pooled 2560-d vector per document (`100k × 2560 × 2 = 512 MB`). Cost: the J-space head loses token-level resolution. **Do not rent GPUs.** Record `"pooling": "mean"` in `artifacts/MANIFEST.json`.

---

## Phase 3 — Heads, calibration, evaluation (Days 5–6, T4)

Notebook `04_heads_and_eval.ipynb`. Freeze the student trunk; train small heads on top. Each head is 1–2 linear layers.

- [ ] **head_lexical** — multi-label over the lexical pattern ids. Unit: span. Target: weak labels. Purpose is *recall beyond the regex*.
- [ ] **head_rhetorical** — multi-label over `patterns.rhetorical.yaml` ids. Unit: span/sentence. Target: spaCy-derived heuristic labels from Phase 1.
- [ ] **head_construction** — regression over `evenness`, `burstiness`, `recap_closure`, `over_explain`, `portability`. Unit: paragraph and piece. StoryScope 30-core is **not** in v1.
- [ ] **head_jspace** — multi-label over the frozen 512-word filtered vocabulary. Unit: sentence. Diagnostic, developer view only in v1.
- [ ] **head_contrast** — binary `pile` with a contrastive objective. Output is **`matches_ai_pile`, never a probability of AI authorship.**
- [ ] **Calibration.** `calibrate.py` computes the `matches_ai_pile` distribution over the held-out FineWeb-Edu human slice. Operating threshold at **1% false-positive rate on human text**. Store `artifacts/calibration.json`.
- [ ] **Evaluation slices and targets:**

| Slice | Metric | Target |
|---|---|---|
| Held-out RAID (unattacked, 5k) | contrast AUC | ≥ 0.90 [inferred from Fast's 0.93] |
| RAID attacked rows | contrast AUC | report honestly; expect a drop, do not tune on it |
| Pangram slice, if obtainable | contrast AUC | report; OOD generalization check |
| FineWeb-Edu human (3k) | FPR at operating threshold | ≤ 1% |
| 200 hand-audited spans | style-hit precision | ≥ 0.85 |
| Synthetic Slopasaurus-like prompts | style-hit recall on planted patterns | ≥ 0.70 |

- [ ] **Style-hit precision audit is manual.** Sample 200 predicted spans stratified by pattern id. If a pattern id scores below 0.6 precision, set `enabled: false`.
- [ ] **Teacher-only diagnostics, never shipped:** Binoculars and DetectGPT on a 1k subset.

**Rollback:** heads are independent. If `head_jspace` produces nothing interpretable, drop it from v1. If `head_construction` is uncorrelated with the deterministic stats, ship the stats directly.

---

## Fail-closed artifact verification

`src/slopdet/verify.py`, called before any inference. `artifacts/MANIFEST.json` pins SHA-256 of student, heads, ontology, jspace vocab, znorm, calibration; records tokenizer, teacher layer, pooling, `trained_on`, `never_trained_on`.

Rules:

- Every hash is recomputed at load. Any mismatch raises, and the caller returns `{"status": "unverified_artifact", "hits": [], "resemblance": null}`.
- It does **not** degrade to regex-only on mismatch.
- Every emitted result carries `artifact_sha256`.
- `never_trained_on` is machine-readable so eval scripts can assert against it.

## Hit schema (frozen for v1)

See `docs/schema/hit-schema.json` (to be written in Phase 0/7): lanes `style` vs `resemblance` never summed; empty hits = "Nothing matched."

## Copy and UI rules (binding on every surface)

- Never "87% AI." Never "AI-generated." Never "written by ChatGPT." Never a grade.
- Resemblance renders as: **"Resembles the AI pile more than 94% of human reference texts."** The comparison class is always named.
- Style hits render as: pattern name, the quoted span, the short fix.
- **Empty hit list renders as "Nothing matched."** Not "This looks human."
- The two lanes are visually separated and never summed into one score.
- Highlight units are enforced by the renderer: style = smallest span, resemblance = sentence, construction = paragraph or piece.

## In v1 vs later

**v1 (this plan):** ontology YAMLs, corpus builder, teacher cache, J-lens fit + filter, distilled student, five heads, calibration, hit schema, SHA-pinned fail-closed loader, CLI, `extension/INTERFACE.md` stub.

**Later, explicitly out:**

- Browser extension and select-text overlay — packaging. v1 freezes the JSON contract only.
- StoryScope long-fiction 30-core head — blocked on missing human story text.
- Binoculars or any two-model scoring in the shipped product.
- Phone (Oppo Reno 10) — inference-only, later, after quantization.
- Academic-integrity positioning.
- Merging the two lanes into one number — permanently out.

## Week sequence

| Day | Work | GPU |
|---|---|---|
| 1 | Phase 0: tree, three YAMLs, `ontology.py`, tests, tag `ontology-v1` | none |
| 2 | Phase 1: downloads, subsample, weak-label, construction stats, splits | ~1h T4 for selfgen |
| 3 | Phase 2a: confirm hidden size, cache 20k×64 residuals in 40 shards | 1–2 T4 sessions |
| 4 | Phase 2b: J-lens fit + 4-stage filter; student distill 20 epochs | 3–5h T4 |
| 5 | Phase 3a: train five heads, calibrate to 1% FPR | ~2h T4 |
| 6 | Phase 3b: eval slices, 200-span manual precision audit, disable weak ids | none |
| 7 | Freeze `MANIFEST.json`, write `extension/INTERFACE.md`, README, notices | none |

## Success criteria

- [ ] `ontology-v1` tagged with ≥70 ids, every one carrying a fix blurb and a license note; CC BY-SA content isolated to one file with attribution.
- [ ] `data/corpus.jsonl` ≈100k docs, four sources, weak spans + construction stats populated, splits disjoint, RAID-test provably absent.
- [ ] Teacher cache completes at 20k×64×2560 fp16 within Drive, resumable across at least one forced disconnect.
- [ ] `jspace_vocab.json` is a closed 512-word list with the kill count logged for each of the four filters.
- [ ] Student trained; dev cosine similarity to teacher residual reported.
- [ ] Contrast AUC ≥ 0.90 on held-out unattacked RAID; FPR ≤ 1% on held-out FineWeb-Edu at the operating threshold.
- [ ] Style-hit precision ≥ 0.85 on the 200-span manual audit, with sub-0.6 pattern ids disabled rather than shipped.
- [ ] `verify.py` returns `unverified_artifact` with an empty hit list on any hash mismatch, proven by a test that corrupts a byte.
- [ ] No output surface anywhere can produce a percentage-of-AI string; a grep test asserts it.
- [ ] `extension/INTERFACE.md` documents the frozen JSON contract, and no extension code exists.

## Risks and mitigations

- **Token-level cache does not fit or does not finish.** Ordered degradation, then mean-pool Fast rollback. Never rent GPUs.
- **Weak labels are noisy.** 200-span manual audit gates shipping; heads evaluated on recall *beyond* the regex.
- **J-space surface confounds.** Four-stage filter plus closed allowlist; droppable in v1.
- **FineWeb-Edu post-2022 AI contamination.** Filter to pre-2022 dumps; report residual risk.
- **Contrast head overfits RAID.** Self-generated non-teacher families; Pangram/attacked-RAID OOD, never tuned on.
- **Product drifts into authorship claims.** Two-lane split enforced in schema, renderer, and a grep test.
