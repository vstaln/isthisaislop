#!/usr/bin/env python3
"""Build notebooks/SlopDetector_Colab.ipynb with the repo files embedded."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "SlopDetector_Colab.ipynb"

EMBED_PATHS = [
    "src/slopdet/__init__.py",
    "src/slopdet/calibrate.py",
    "src/slopdet/span.py",
    "src/slopdet/construction.py",
    "src/slopdet/ontology.py",
    "src/slopdet/weaklabel.py",
    "src/slopdet/report.py",
    "src/slopdet/scorer.py",
    "src/slopdet/explain.py",
    "src/slopdet/labels.py",
    "src/slopdet/lfm.py",
    "ontology/schema.json",
    "ontology/patterns.core.yaml",
    "ontology/patterns.wikipedia.yaml",
    "ontology/patterns.rhetorical.yaml",
    "ontology/patterns.slop.yaml",
    "scripts/fine_tune_lfm.py",
]


def nb_cell(cell_type: str, source: str, **meta) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell = {"cell_type": cell_type, "metadata": meta, "source": lines}
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def main() -> None:
    files = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in EMBED_PATHS}
    payload = json.dumps(files)

    cells = [
        nb_cell(
            "markdown",
            """# Is This AI Slop? (ITAIS) — train the LFM2.5 encoder

**Runtime → Change runtime type → T4 GPU.** Then **Runtime → Run all**.

Trains one model that does both jobs:

1. **Classify** the text: slop or not slop.
2. **Why:** which sentences + spans, with named patterns.

Base: `LiquidAI/LFM2.5-Encoder-350M` (bidirectional masked-LM encoder, ~354M params),
fine-tuned 1 epoch on the 122k-doc mixed corpus (coai / storyscope / gutenberg / blogs / scp).
fp16 on T4; NaN preflight aborts loudly rather than silently producing garbage.

Exports `artifacts/lfm/` onto Google Drive `MyDrive/isthisaislop/`.
""",
        ),
        nb_cell(
            "code",
            """# Config
EPOCHS = 1
MAX_LEN = 512
MODEL = "LiquidAI/LFM2.5-Encoder-350M"
DATA_PARQUET = "train_all.parquet"   # name of the training parquet (Drive or upload)
DRIVE_PATH = "isthisaislop"          # folder under MyDrive
MOUNT_DRIVE = True

import os, sys
from pathlib import Path

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    drive = None

if IN_COLAB and MOUNT_DRIVE:
    try:
        drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive") / DRIVE_PATH
    except Exception as exc:
        print("Drive mount failed, using /content:", exc)
        ROOT = Path("/content/isthisaislop")
elif IN_COLAB:
    ROOT = Path("/content/isthisaislop")
else:
    ROOT = Path(".").resolve()
    if not (ROOT / "src" / "slopdet").exists():
        ROOT = Path("/home/vstaln/slop-detector")

ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
print("ROOT", ROOT)

import torch
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
else:
    raise SystemExit("No GPU. Runtime → Change runtime type → T4 GPU → Save → Run all.")

import subprocess
pkgs = ["pyyaml", "regex", "jsonschema", "scikit-learn", "numpy>=1.26", "pandas", "pyarrow"]
pkgs += ["transformers", "accelerate", "safetensors"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
print("deps ok")
""",
        ),
        nb_cell(
            "code",
            f"""# Write the package onto disk (self-contained; no git clone required)
import json
from pathlib import Path
FILES = json.loads({payload!r})
for rel, content in FILES.items():
    path = Path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
print("wrote", len(FILES), "files")
import sys
sys.path.insert(0, str(Path("src").resolve()))
print("ITAIS ready")
""",
        ),
        nb_cell(
            "code",
            """# Locate the training parquet: Drive → /content → manual upload
from pathlib import Path

DATA_PARQUET = "train_all.parquet"
candidates = [
    Path(".") / DATA_PARQUET,                     # already in ROOT
    Path("/content") / DATA_PARQUET,              # uploaded to /content
    Path("/content/drive/MyDrive/isthisaislop") / DATA_PARQUET,
]
src = next((p for p in candidates if p.exists()), None)
if src is None:
    raise SystemExit(
        f"train_all.parquet not found. Upload it to Drive {DRIVE_PATH}/ or /content/, "
        f"or re-run the build locally and copy it."
    )
if src.resolve() != (Path(".") / DATA_PARQUET).resolve():
    import shutil
    shutil.copy(src, Path(".") / DATA_PARQUET)
    print(f"copied {src} → ./{DATA_PARQUET}")
else:
    print(f"using ./{DATA_PARQUET}")
""",
        ),
        nb_cell(
            "code",
            """# Train: LFM2.5-Encoder-350M, 1 epoch; fp16 with automatic fp32 fallback
import sys, subprocess
from pathlib import Path

def run_training(extra: list[str]) -> int:
    cmd = [
        sys.executable, "scripts/fine_tune_lfm.py",
        "--arch", "encoder",
        "--model", MODEL,
        "--spans-parquet", DATA_PARQUET,
        "--max-len", str(MAX_LEN),
        "--epochs", str(EPOCHS),
        "--out", "artifacts/lfm",
    ] + extra
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)

# Try fp16 first (default). On Turing T4, fp16 can NaN-abort; fall back to fp32.
rc = run_training([])
if rc != 0:
    print("\\n[retry] fp16 run failed (rc=%d). Retrying with --precision fp32..." % rc)
    rc = run_training(["--precision", "fp32"])
sys.exit(rc)
""",
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
