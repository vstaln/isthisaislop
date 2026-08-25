"""Loading the training corpus — local parquet if present, else the public HF copy.

`data/` is gitignored, so a fresh CUDA box has no corpus. Rather than making the
operator fetch 287 MB by hand before the one training command, `resolve` falls
back to the public dataset (`vstalingrady/itais`, no token needed) and prints
which copy it actually used.

Kept free of torch so the schema and label handling stay unit-testable.
"""

from __future__ import annotations

import json
from pathlib import Path

from .labels import parse_label

HF_DATASET_REPO = "vstalingrady/itais"
V2_TRAIN_FILE = "v2_train_labeled.parquet"
V2_HOLDOUT_FILES = (
    "v2_holdout_labeled.parquet",
    "v2_holdout_paraphrase_labeled.parquet",
    "v2_holdout_mixed.parquet",
    "v2_holdout_unseen_model_labeled.parquet",
)
# The v2 build wrote this name locally before the HF upload normalised it.
LEGACY_LOCAL_NAMES = ("v2_train.parquet.labeled.parquet", "v2_train.labeled.parquet")

WANTED_COLUMNS = ("text", "label", "pile", "register", "spans", "split_hint")


def resolve(path: Path, hf_file: str = V2_TRAIN_FILE, repo: str = HF_DATASET_REPO) -> Path:
    """Return a readable parquet path, downloading from HF only when needed."""
    if path.exists():
        print(f"[corpus] local {path}", flush=True)
        return path
    for legacy in LEGACY_LOCAL_NAMES:
        candidate = path.with_name(legacy)
        if candidate.exists():
            print(f"[corpus] local {candidate} (legacy name)", flush=True)
            return candidate
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            f"no corpus at {path} and huggingface_hub is not installed. Either place the "
            f"parquet there or install the train extra to pull {repo}/{hf_file}."
        ) from exc
    print(f"[corpus] {path} missing — downloading {repo}/{hf_file}", flush=True)
    cached = hf_hub_download(repo_id=repo, filename=hf_file, repo_type="dataset")
    print(f"[corpus] using {cached}", flush=True)
    return Path(cached)


def coerce_spans(spans) -> list[dict]:
    """Span dicts that carry a lane, tolerating None / str / list / ndarray."""
    if spans is None:
        return []
    if isinstance(spans, str):
        try:
            spans = json.loads(spans)
        except json.JSONDecodeError:
            return []
    try:
        return [s for s in spans if isinstance(s, dict) and s.get("lane")]
    except TypeError:
        return []


def smoke_rows(n: int = 24) -> list[dict]:
    """Synthetic rows that exercise the graph without a corpus: two registers, one
    lane, and pair keys so the split and the contrastive sampler are covered too."""
    slop = "Here's the thing: we leverage robust pipelines to unlock synergies. "
    human = "Thursday mornings at the clinic were empty, so I counted 41 chairs. "
    rows = []
    for i in range(n):
        rows.append({"text": slop * 3, "label": 1, "register": "smoke",
                     "spans": [{"lane": "glue", "start": 25, "end": 33}],
                     "split_hint": f"fictpair:{i}"})
        rows.append({"text": human * 3, "label": 0, "register": "smoke",
                     "spans": [], "split_hint": f"fictpair:{i}"})
    return rows


def load_rows(parquet: Path, chunk: int = 10_000) -> list[dict]:
    """Stream the corpus into row dicts: text, label, register, spans, split_hint.

    Read in slices so peak RAM stays bounded on a small box. Missing labels and
    unknown registers raise here rather than silently training on garbage.
    """
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(parquet).names)
    if "text" not in schema_names:
        raise SystemExit(f"{parquet}: missing 'text' column")
    if "register" not in schema_names:
        raise SystemExit(
            f"{parquet}: missing 'register' column — per-register calibration needs it; "
            "rebuild with scripts/build_training_parquet.py"
        )
    columns = [c for c in WANTED_COLUMNS if c in schema_names]

    rows: list[dict] = []
    reader = pq.ParquetFile(parquet)
    for batch in reader.iter_batches(batch_size=chunk, columns=columns):
        for rec in batch.to_pylist():
            register = rec.get("register")
            if not register:
                raise SystemExit(f"row missing register: {rec.get('text', '')[:60]!r}")
            if rec.get("label") is None and rec.get("pile") is None:
                raise SystemExit(f"row missing label/pile: register={register}")
            rows.append({
                "text": rec["text"],
                "label": parse_label(rec, default=0),
                "register": register,
                "spans": coerce_spans(rec.get("spans")),
                "split_hint": rec.get("split_hint") or "",
            })
    return rows


def check_registers(rows: list[dict]) -> None:
    """Fail before training on a register the calibration registry does not know."""
    from .calibrate import register_allowed

    unknown = sorted({r["register"] for r in rows if not register_allowed(r["register"])})
    if unknown:
        raise SystemExit(
            f"unknown register(s) {unknown} — add them to slopdet.calibrate.ALLOWED_REGISTERS "
            "or fix the data build"
        )
    bad = sorted({r["label"] for r in rows} - {0, 1})
    if bad:
        raise SystemExit(f"bad label values {bad}")
