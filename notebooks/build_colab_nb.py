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
    "ontology/schema.json",
    "ontology/patterns.core.yaml",
    "ontology/patterns.wikipedia.yaml",
    "ontology/patterns.rhetorical.yaml",
    "ontology/patterns.slop.yaml",
    "notebooks/colab_pipeline.py",
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
            """# Is This AI Slop? (ITAIS) — classify + why

**Runtime → Change runtime type → T4 GPU.** Then **Runtime → Run all.**

Trains one local model that does both jobs:

1. **Classify** the text: slop or not slop.
2. **Why:** which sentences, plus named patterns (`leverage`, `here's the thing`, …) with a short fix.

Do not rent a GPU. A free Colab T4 is enough (~20–40 min, 1 epoch, `roberta-base` 125M).

Data: coai (arxiv abstracts vs claude-haiku-4.5 / gemini-3-flash / gpt-oss-120b / gpt-5-nano paraphrases), stitched so the model sees mixed documents and learns *which span* is which.

Exports `artifacts/roberta-span/` onto Google Drive `MyDrive/isthisaislop/`.
""",
        ),
        nb_cell(
            "code",
            """# Config
EPOCHS = 1
MAX_LEN = 256
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
        ROOT = Path("/content/drive/MyDrive/isthisaislop")
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
            """# Train: classify + mark which sentences. Demo prints named why on each span.
from pathlib import Path
import sys
sys.path.insert(0, str(Path("src").resolve()))
sys.path.insert(0, str(Path("notebooks").resolve()))
from colab_pipeline import train_span_roberta, demo_span

info = train_span_roberta(Path(".").resolve(), epochs=EPOCHS, max_len=MAX_LEN)
demo_span(Path(".").resolve(), info)
print("done — bundle at", Path("artifacts/roberta-span").resolve())
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
