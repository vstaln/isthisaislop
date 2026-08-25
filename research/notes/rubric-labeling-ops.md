# Rubric labeling ops — throughput, order, QC, cost (250k docs)

Planning note. All rate-limit numbers are ASSUMPTIONS (not measured on stealth/ox-alpha):
OpenRouter free-tier models commonly cap ~20 req/min/model (~1,200 calls/hr). Measured prior
lanes: gemma ~1 doc/2min (thought-token burn), laguna ~1 doc/min sequential with 503 storms
(`llm-labeling-efficiency.md`). mimo-v2.5 is the only measured-quality lane (69.2% acc vs pile).

## 1. Throughput math

Assumptions: 3–8 s/call sustained (midpoint 5 s used), 1 doc/call unless batched, no backoff
storms. "Capped" = the ~20 req/min free-tier ceiling dominates regardless of worker count.
Batched N=4 docs/call amortizes latency and 4×'s effective docs/hr within the same call cap.

| Workers | docs/hr @5s/call | 250k | 100k | 50k | 10k |
|---|---|---|---|---|---|
| 8 (uncapped) | 5,760 | 43 h | 17 h | 8.7 h | 1.7 h |
| 16 (uncapped) | 11,520 | 22 h | 8.7 h | 4.3 h | 0.9 h |
| 32 (uncapped) | 23,040 | 11 h | 4.3 h | 2.2 h | 0.4 h |
| capped 20 rpm | 1,200 | 208 h (~8.7 d) | 83 h | 42 h | 8.3 h |
| capped + batch N=4 | 4,800 | 52 h | 21 h | 10.4 h | 2.1 h |

Read: worker count beyond ~8 is pointless if the cap holds; **batching (N=4) is the real lever**,
consistent with P1 in `llm-labeling-efficiency.md`. Realistic risk: free tiers also throttle by
tokens/day and burst; assume ±50% on all capped figures and expect 503/retry storms to add
10–30%. Full 250k on one free model is a multi-day commitment even batched — argues for
subsamples per register (Phase 2), not census.

## 2. Labeling priority: where does the rubric earn its keep?

Rubric value = narrative/structural signal (pacing, dialogue rhythm, trope density, scene
structure) that regex cannot see. Deterministic regex already covers lexical tics well in casual/
formulaic web prose.

| Register | n | Rubric value | Why |
|---|---|---|---|
| storyscope | 50k AI fic | HIGH | AI fiction: structural tells are the whole signal |
| writingprompts | 100k human fic | HIGH | human fiction counterpart; rubric contrast pair |
| gutenberg | 15k human | HIGH | long-form human prose anchor for fiction axis |
| RAID | 35k multi | MED | adversarial/multi-domain; regex-resistant slices |
| M4 | 40k multi | MED | domain diversity; subsample enough |
| coai | 62k paired acad | LOW | deterministic lane already labels it (spans done) |
| wiki_intro | 30k paired | LOW | formulaic structure regex handles |
| HC3 | 18k qa | LOW | short qa prose; lexical tics dominate |
| blogs | 8k | LOW | casual web prose = regex home turf |
| beemo | 4k | LOW | small; use as eval slice only |

**Phase 1 (first 10k, highest value): storyscope 4k + writingprompts 4k + gutenberg 2k.**
Fiction axis first — it is where the rubric adds irreplaceable signal and where the deterministic
lane is weakest. At capped+batched 4,800 docs/hr this is ~2 h of wall clock.

**Phase 2 order:** M4 subsample 20k → wiki_intro subsample 10k → RAID full 35k → HC3 5k →
blogs 3k → beemo 2k → coai spot-check slice 2k (regex already covers the rest).
Phase 2 total ≈ 77k ≈ 16 h batched-capped. Skip full coai/wiki/HC3 LLM passes entirely;
deterministic coverage there is the point of those registers.

## 3. Quality control protocol

- **Self-consistency:** every 25th doc re-scored at temp 0.7 (independent call, same prompt).
  Track label-match rate over rolling 200-doc windows. Gate: ≥80% self-agreement or pause lane.
- **Provenance agreement:** paired registers (coai, M4, wiki_intro, RAID, beemo) carry known
  ground-truth labels. Compute accuracy vs provenance on every register touched. Gate: ≥70%
  (mimo's measured 69–70% is the floor precedent); below gate → probe alternate model on 20 docs
  before continuing (pattern from the mimo-vs-deepseek reversal: deepseek hit 64.9% + 42%
  cross-lane agreement and was demoted).
- **Cross-check:** 50-doc overlap scored by both self-consistency pass and provenance calc;
  divergence between the two signals flags prompt drift before it poisons a register.
- All gates logged per-register into the jsonl `_usage`/meta so a bad window can be re-run
  (`already_done()` dedupe means re-runs need explicit invalidation).

## 4. Cost ceiling (VM time at $0.105/hr, API $0)

| Scenario | Wall clock (250k) | VM cost | Note (10k Phase 1) |
|---|---|---|---|
| Uncapped 32 workers | 11 h | $1.16 | 0.4 h / $0.04 |
| Capped 20 rpm, 1-doc | 208 h | $21.84 | 8.3 h / $0.87 |
| Capped + batch N=4 | 52 h | $5.46 | 2.1 h / $0.22 |

Instance-time cost is negligible either way (<$22 worst case); the binding constraint is calendar
time and free-tier token caps, not dollars. Concurrent-with-nothing-else assumption: labeling is
network-bound, CPU idle, so co-tenancy with training would not change these materially.

## Bottom line

Batch N=4 on the fastest non-reasoning free lane, cap ~8 workers (more is wasted under the rpm
cap), label fiction first (storyscope/writingprompts/gutenberg = 10k, ~2 h), gate every register
on ≥80% self-consistency + ≥70% provenance accuracy, and treat full-corpus LLM passes on
coai/wiki/HC3/blogs as unnecessary — regex owns those.
