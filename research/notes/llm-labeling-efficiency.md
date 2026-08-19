# LLM labeling lane efficiency — findings and recommendation

Scope: make the two LLM labeling lanes (`scripts/label_gemini.py`, `scripts/label_laguna.py`)
dramatically faster and cheaper. Evidence: read both scripts, `docs/HANDOFF.md` (§3, §6, STATUS,
NEXT ACTIONS), `/tmp/laguna_s21.log` (52 lines, 14/130 docs), `/tmp/gemma_on_laguna.log`
(5 lines, startup only), usage payloads in `/tmp/laguna_s21.jsonl` and `/tmp/gemma_on_laguna.jsonl`.

## 1. Current state (measured)

| Lane | Model / endpoint | Pace (measured) | Tokens / doc | Bottleneck |
|---|---|---|---|---|
| gemini | `gemma-4-31b-it` @ generativelanguage, 4 workers, 1 call/doc | ~1 doc / 2 min (task log); 20–95 s/call | ~400–600 in + **500–900 thought** + ~30 out | reasoning burn; `thinkingConfig` cannot be disabled (HTTP 400); latency per call |
| laguna | `poolside/laguna-s-2.1-free` @ commandcode, **sequential**, curl/call, 2 s sleep, 12 retries, backoff `min(45, 10+a*10)` s | ~1 doc / min (14/130 in log) | ~360 in + ~35 out, `reasoning_tokens: 0` | **503 overloaded_error storms** (every doc retries 1–5×, each retry adds 10–45 s); no concurrency |
| deterministic | `label_coai_batch.py` | fast, multiprocess | n/a | not the problem |

Measured facts from the log: laguna free tier 503s on roughly every other doc (attempts 1–5 per doc
before success); doc 15 was still retrying at log end. `completion_tokens_details.reasoning_tokens`
is 0 for laguna-s-2.1 — it is NOT burning thinking tokens (the gemma lane's main waste does not
apply here). `already_done()` dedupe and per-doc append already exist in both lanes; nothing is
re-labeled on resume, so caching is done.

Key context from HANDOFF.md: **LLM lanes are validation, not the training-data source.** The neural
model's token head is trained on deterministic lane labels (`spans_*.parquet`, ontology + construction
vocabulary, §3.2). LLM labels build/check the eval set. So the LLM lane's job is: produce a
per-register labeled eval slice with *reasonable* quality — throughput is a means, and a slightly
smaller sample is defensible, but don't gut it: §5 requires per-register evaluation and the eval set
is the only AI-vs-human ground truth we have. Also: `.env` already carries OPENROUTER_API_KEY
(default `google/gemma-4-26b-a4b-it:free`), TOKENROUTER_API_KEY (default `deepseek/deepseek-v4-pro-0813-free`),
and COMMAND_CODE_MODEL (currently `poolside/laguna-s-2.1-free`). All three are free-tier keys, so
cost is dominated by token burn and retry waste, not per-token price.

## 2. Ranked proposals

### P1 — Batch N docs per LLM call (JSON array), both lanes. Effort: medium. Speedup: 4–8×.

One HTTP call labels N docs; the model returns a JSON **array** of the existing single-doc schema.
- gemma: 20–95 s/call becomes 20–95 s per N docs → N=4 gives ~4–8× wall-clock on 130 docs
  (~4.3 h → ~40–70 min; N=8 → ~25–45 min, but see risk). **This is the single biggest gemma win.**
  Token spend does NOT shrink — thought tokens scale with N — but latency and per-call overhead
  amortize.
- laguna: same latency amortization; also cuts the number of 503 lottery tickets per doc.
- Changes: extend `SYS` with "return a JSON array, one object per doc, same schema per element";
  parse array, `hydrate_lanes()` + `verbatim()` per element; append per element; keep the existing
  empty-lane/bad-JSON nudges but apply per element.
- Risks (medium): schema breakage (array vs object contract); one bad element corrupting the batch;
  output token cap — gemma `maxOutputTokens: 1200` must grow to ~N×200 + thinking headroom (set
  8192 for N=8), laguna `max_tokens` likewise; reasoning models may cap thought length on long
  batches. **Mitigation: on any batch failure, fall back to 1-doc/call for the failed docs only.**
  `verbatim()` already kills fabricated quotes, so quality per element is protected.
- Note: text truncation stays at 6000 chars/doc; 8×6000 ≈ 12–16k input tokens — fine on all the
  models in play.

### P2 — Switch the laguna lane off the 503-storm model to a fast non-reasoning model on the same gateway. Effort: trivial (env var). Speedup: 10–30×.

The bottleneck is the free tier, not the code. Options, in order:
- **`mimo-v2.5`** on commandcode — MEASURED 70% acc on the 130 eval docs, fast, no 503s. Real
  quality floor already in hand; it is the known quantity.
- **`deepseek-v4-flash`** on commandcode — flash tier, and `label_laguna.py` already sends
  `"thinking": {"type": "disabled"}` for any `deepseek` model (verified in code), so no thought burn
  and no 503 storm expected.
- `mimo-v2.5-pro` / `kimi-k3` as higher-quality fallbacks if the fast ones underperform.
- Also available on the same key: 56 models incl. `glm-5.2`, `minimax-m3`, `gpt-5.x`, `claude-*`.
All changes are `COMMAND_CODE_MODEL=...` in `.env` — the script is already env-driven.
- **Expected result: ~10–30 docs/min at 4–5 workers instead of ~1 doc/min** → 130 docs in ~5–15 min.
- Risk: quality regression vs laguna-s-2.1 — but laguna-s-2.1 quality is **unmeasured** (the 14
  labeled docs in the log are seed docs, not eval docs). "Regression" is hypothetical against a
  model that cannot complete the eval set. If laguna quality matters, run a 20-doc laguna-vs-mimo
  spot check on OpenRouter (`poolside/laguna-s-2.1` exists there per earlier check) — one .env var.

### P3 — Concurrency + remove the 2 s sleep + retry tuning on laguna. Effort: small. Speedup: ~2–4× (only if the model stops 503ing; see P2).

- `ThreadPoolExecutor` like the gemini script (the pattern already exists in-repo, copy it), capped
  at 3–5 workers so a rate-limited free tier doesn't get hammered into bans.
- Delete the unconditional `time.sleep(2)` between docs (saves 2 s × 130 = ~4 min of pure sleep).
- Retry: honor `Retry-After` if the gateway sends it; keep exponential backoff but with jitter and a
  cap aligned to the observed throttle (~60 s). Current `min(45, 10+a*10)` is fine; add jitter.
- Risk: moderate — concurrency on an overloaded free tier can worsen 503s and, in the worst case,
  get the key throttled. This is why P2 (pick a model that doesn't 503) is ranked above P3.

### P4 — Drop the gemma reasoning model for a non-reasoning one. Effort: small (one probe). Speedup: 2–3× token + latency; enables cheaper batching.

`gemma-4-31b-it` burns 500–900 thought tokens/doc and refuses `thinkingConfig: disabled`. But:
- `gemma-4-26b-a4b-it` is already the script's ALT model and is the **OpenRouter default** in
  `label_laguna.py` (`google/gemma-4-26b-a4b-it:free`) — so the laguna code path (which handles
  list/`</think>` stripping and is not the Gemini API) can run gemma-26b at near-zero thought burn.
- Action: one 5-min probe — run `call_gemini("gemma-4-26b-a4b-it", ...)` on 5 docs and read
  `thoughtsTokenCount`. If ≈0, make it the default for the gemini lane (and/or route gemma-26b
  through the OpenRouter key using `label_laguna.py`, which needs zero new code).
- This also cuts the per-call output budget needed for P1 batching (no thinking headroom).

### P5 — Right-size the eval pass instead of full-LLM-ing everything. Effort: trivial. Speedup: 2–3× at the source.

The handoff says LLM labels are validation. A per-register stratified sample of ~60–100 docs
(coai, fiction, blog, storyscope) is enough to measure the deterministic lane and gate the neural
model — the current 130 is already a sample, not a census. **Counterargument (why not to shrink
aggressively):** the eval set is the only AI-vs-human ground truth; shrinking below ~50 raises
measurement variance on per-register gates. So: keep 100–130 but make it fast via P1/P2; shrink to
60 only if the lane is still too slow after P1–P4.

### P6 (minor) — Record `_usage` into the jsonl for gemma. Effort: 5 min.

`label_gemini.py` prints usage but does not persist it (log shows `None` in records); laguna does
persist. Persisting lets future runs cost-compare models. Not a throughput fix.

## 3. Recommended target pipeline

1. **Model:** `deepseek-v4-flash` on commandcode (thinking disabled by the existing code path), with
   `mimo-v2.5` as the drop-in fallback if flash quality disappoints. Keep laguna-s-2.1 reachable via
   OpenRouter for a 20-doc spot check, but do not run the free commandcode tier again.
2. **Concurrency:** 5 workers, `ThreadPoolExecutor` (copy the gemini worker pattern), jittered
   exponential backoff capped ~60 s, `Retry-After` honored, **no fixed sleep**.
3. **Batching:** N=4 docs/call, JSON-array schema, per-element `hydrate_lanes` + `verbatim`, 1-doc
   fallback on batch failure; `max_tokens` ~4096.
4. **Gemini lane:** probe `gemma-4-26b-a4b-it` for thought burn (P4); if it's clean, prefer it or
   route gemma-26b through OpenRouter — keep gemma-4-31b-it only if quality is measurably better.
5. **Expected result:** 130 docs in ~10–20 min (≈7–13 docs/min) vs the current ~4.5 h combined —
   **~15–25× end-to-end**, near-zero cost on free tiers, one committed artifact
   (`eval/labels/laguna.jsonl` / `gemma.jsonl`).

## 4. Decision: keep laguna at all?

**Keep the lane, drop the model.** The laguna *code path* is the good one — it's the base for the
gemini script, has resume, nudges, verbatim filtering, and a sensible schema. `laguna-s-2.1-free`
on commandcode is the problem: unmeasured quality + a free tier that 503s every other call. It costs
nothing to keep as a `.env` value; run it only for a 20-doc quality spot check against `mimo-v2.5`.
If mimo's measured 70% is acceptable (it's the only measured floor we have), laguna-s-2.1 stays
retired until a paid/OpenRouter tier shows it beats that number.

## 5. Implementation order for the next agent

1. `P2` first (5 min): set `COMMAND_CODE_MODEL=mimo-v2.5` (or deepseek-v4-flash), re-run the 130 —
   this alone fixes the day.
2. `P3` (30–60 min): port the gemini worker/ThreadPoolExecutor pattern into `label_laguna.py`, drop
   the 2 s sleep, jitter backoff.
3. `P1` batching on `label_laguna.py` (1–2 h): array schema + per-element hydrate + fallback; then
   port to `label_gemini.py`.
4. `P4` probe (10 min) while batching is being written.
5. Re-run `python -m pytest -q` (31 pass baseline) after each change — the schema contract
   (`hydrate_lanes`) is load-bearing and tested.

## 6. Honest risk register

- Batch schema change can break the SYS-prompt contract → fallback-to-single is mandatory, and
  `verbatim()` (quote must literally appear in text) already protects against fabrication.
- Concurrency on free tiers risks key throttling → cap workers at 5, use jitter; P2 reduces the
  trigger surface because the fast models don't 503.
- Thought-token cap on gemma batches is unverified → probe before committing to N=8; N=4 is the safe
  start.
- 70% (mimo) vs unmeasured (laguna) is a weak quality comparison — the 20-doc spot check is cheap
  insurance, but do not block the throughput fix on it.

## Update (measured, post-implementation)

Ran the 130-doc eval on **deepseek-v4-flash** (P3 5-worker concurrent run, ~12 min
vs ~4.5h for gemma/2h for laguna-free):

| Lane | acc vs pile | slop/human/mixed/unclear | notes |
|---|---|---|---|
| mimo-v2.5 (eval/labels/laguna.jsonl) | 69.2% (117) | 68/47/2/11 | best; matches pile dist (60H/70A) |
| deepseek-v4-flash (/tmp/deepseek_on_eval.jsonl) | 64.9% (114) | 21/72/21/16 | human-biased; under-detects slop |
| pairwise agreement mimo vs deepseek | 42% | | lanes disagree on most docs |

**Verdict: mimo-v2.5 stays the primary LLM lane.** deepseek-v4-flash is faster
(~10-30x) but -4.3pts acc + heavy human bias (21 vs 68 slop calls) + only 42%
agreement with mimo. P2's "switch to deepseek" recommendation is REVERSED:
keep mimo for quality, use deepseek only for speed when quality is secondary.

P4 (gemma-4-26b-a4b-it thought-burn probe): **DEAD** — 26b burns 753 thought
tokens/doc, same as 31b, and 2/3 probe docs failed ("empty answer"). Both gemma
models are unusably slow for labeling; P1 batching (N docs/call) is the only
gemma lever and is not worth it if mimo+deepseek cover the lanes.
