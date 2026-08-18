# HANDOFF — ITAIS slop detector: fine-tune setup

> Written 2026-08-18 for the next model to pick up. Goal: fine-tune a modern LLM
> into an AI-slop detector using the data built in this repo, then evaluate.

## What this project is

Select text → verdict + why. Two layers:
1. **Doc verdict**: "matches AI pile vs human" — a calibrated classifier, never an authorship claim, never a %-AI number.
2. **Why**: checkable hits — verbatim quoted span + tag id + hardcoded fix (e.g. `glue` → "leverage", say: "Name the action. Use a concrete verb.").

## Model decision (already made, user-approved)

- **Base**: `LiquidAI/LFM2.5-1.2B` (base variant, NOT `-Instruct`). 2026-gen, linear-time attention → 5k-word stories fit cheaply. License LFM Open v1.0: free commercial use below $10M annual revenue; Apache-2.0 alternative if that ever matters: `Qwen/Qwen3-0.6B`.
- **Method**: `AutoModelForSequenceClassification(num_labels=2)` — LM head swapped for a 2-class head over last-token hidden state. LoRA (`r=16, lora_alpha=32, target_modules="all-linear"`), 1 epoch, fp16, `use_cache=False`, max_length 2048. Trainer + eval AUC + calibrate 1% FPR on human test slice (existing `slopdet.calibrate.calibration_record`).
- **Data**: coai 62,460 train docs (12M tokens) + sampled StoryScope AI fiction (~20M tokens) + WritingPrompts/SCP/blogs human samples (~15M) ≈ **50M tokens, 1 epoch**. Eval on coai test (download via `notebooks/colab_pipeline.py:download_coai`).
- **Span model** (phase 2): same script with `AutoModelForTokenClassification`, labels from the spans parquet below.
- **Why 1 epoch**: fine-tune not pretrain; overfit risk > signal after epoch 1. Look at val curve; extend only if still climbing. Grokking doesn't occur in this regime.

## Data assets (all on disk)

| Path | Contents |
|---|---|
| `data/coai_train.parquet` | 62,460 docs, cols `text,label,model_name` (31,230 AI / 31,230 human) |
| `data/training/spans_coai_train.parquet` | **The labeled corpus**: same 62,460 rows, cols `text,pile,slop_tags,human_tags,spans` (JSON: id/lane/lean/start/end/quote). 172,169 spans, 0 non-verbatim. |
| `data/training/labeled.parquet` | 250 eval docs (130 AI / 120 human) with LLM lean + local spans |
| `data/raw/storyscope/` | stories_dev/val/test/train.parquet — 5 AI stories per prompt row (`story_gpt`, `story_claude`, `story_gemini`, `story_deepseek`, `story_kimi`), GPT-5.4/DeepSeek V3.2/Kimi K2.5/Gemini 3 Flash/Claude 4.6. Human text excluded (copyright) — Gutenberg covers that side. |
| `data/raw/gutenberg_fiction/` | 4 shards `train-*.parquet`, cols `file_id,text_sub_id,text,tokens` — human fiction, public domain (~393k chunks) |
| `data/raw/blogs/` | `blogs/` extracted — 19,320 files, Schler blog corpus (human, the actual slop register). **Needs text extraction from HTML/XML before use** |
| `data/raw/writingprompts/` | train-0/1.parquet + test.parquet (~600MB) — modern human short fiction, Reddit user content |
| `data/raw/scp/` | scp_tales + stories1-7 jsonl (~116MB) — CC-BY-SA modern fiction |
| `data/raw/manifest.json` | file registry with licenses |

## Labeling system (the 3 lanes — all deterministic, checkable)

- **style** — 415 ontology regexes (`ontology/*.yaml`, loaded by `slopdet.ontology.load_ontology`). Glue/puffery/frames/cliche/weasel… Sources: antislop sampler + Kobak 2406.07016.
- **construction** — heuristics in `slopdet.construction` + `explain._human_signals`: recap/gloss/even/moral/jump/anchor/burst/weekday/clock/number/name/spoken/first/contrast.
- **storyscope** — `slopdet.storyscope.py`, 8 checks from arXiv:2604.03136: moralize/sensory/causal/realize/intro/agency (slop) + reader/dialogue (human).

**Invariants (do not break):**
- Every hit carries a **verbatim quote** — spans with empty or fabricated quotes are bugs (we fixed `gloss` and `name` this session).
- `lean` ∈ {slop, human, mixed, unclear} — evidence balance, NOT a classifier verdict. Human cues (number/name/burst) are **non-discriminative on academic prose** — don't let them drive lean.
- Copy rules: never "% AI", never "AI-generated", never "written by ChatGPT", never authorship claims. Enforced by `slopdet.report.FORBIDDEN_SUBSTRINGS` + tests.
- Tests: `uv run pytest -q` → 31 pass.

## Scripts

- `scripts/label_coai_batch.py` — blasts all 62k coai docs through `explain(sentences=False)` with multiprocessing (~3 min on 8 cores). Model for the span-label blast of StoryScope/Gutenberg/etc.
- `scripts/export_training.py` — builds `data/training/labeled.parquet` from eval/labels/*.jsonl; backfills `explain()` when `local` missing.
- `scripts/label_laguna.py` — optional LLM lane (mimo/laguna via OpenCode Go). Records `laguna` lean + verified quotes. Verbatim filter + format nudges built in. Don't rely on it for training labels; the deterministic lane is the data source.
- `scripts/fetch_datasets.py` — dataset downloader, manifest writer. Jobs: storyscope, gutenberg_fiction, blogs, writingprompts, scp.
- `scripts/train_cpu_scorer.py` — CPU logistic baseline on coai features (already trained → `artifacts/sklearn_bundle.json`).
- `notebooks/colab_pipeline.py` — `fine_tune_roberta`/`train_span_roberta` (the OLD plan, roberta-base). LFM supersedes roberta for the doc model; keep span trainer as reference.

## API config

- `.env` (gitignored): `COMMAND_CODE_*` = **OpenCode Go** gateway: base `https://opencode.ai/zen/go/v1`, model `mimo-v2.5`, key `sk-…` (also `TOKENROUTER_*`, `OPENROUTER_*` present).
- Free-tier models (`*-free` on `https://opencode.ai/zen/v1`) are rate-limited/saturated — use Go endpoint for batch LLM labeling.
- The key is also in `~/.local/share/opencode/auth.json` under `opencode-go`.

## Gotchas learned this session

- `pkill -f <pattern>` matches the calling shell's own command line → self-kill. Use bracket trick: `pkill -f 'label_lagun[a]'`.
- Long-running jobs: `setsid nohup … < /dev/null & disown` or the shell teardown kills them.
- `explain()` is slow cold (~0.7s) — `lru_cache` on `load_ontology` and `scorer.load_bundle` fixed it; `sentences=False` skips the per-sentence pass.
- coai AI docs are arxiv paraphrases — academic register, low slop density. Fiction slop lives in StoryScope/WritingPrompts; the storyscope lane is untested at scale — blast StoryScope next.
- Blogs are raw HTML — extract text first.
- `data/` and `artifacts/` are gitignored; eval labels and code are committed-able but **nothing was committed this session** (uncommitted changes exist — review `git status`).

## Next steps (for the other model)

1. Write `scripts/fine_tune_lfm.py` (the inline example given in this session's transcript is the spec) or port `fine_tune_roberta` — needs torch/transformers/peft on the rented GPU box. `uv add --dev` nothing; keep deps in the script's install notes.
2. Blast StoryScope AI stories through the labeler (fiction spans) → `data/training/spans_storyscope.parquet`; sample Gutenberg/WP/SCP/blogs for the human fiction/blogs pile.
3. Train doc model on mixed 50M tokens, eval AUC ≥ 0.95 / FPR ≤ 1% on coai test. Export `artifacts/lfm/` (merged LoRA + tokenizer + calibration.json + manifest.json).
4. Span head (`AutoModelForTokenClassification`) on the spans parquets.
5. Push artifacts to HF repo `vstaln/isthisaislop-*` after training.
