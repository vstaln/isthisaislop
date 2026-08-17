#!/usr/bin/env python3
"""Build notebooks/SlopDetector_Colab.ipynb with the repo files embedded."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "SlopDetector_Colab.ipynb"

EMBED_PATHS = [
    "pyproject.toml",
    "ontology/schema.json",
    "ontology/patterns.core.yaml",
    "ontology/patterns.wikipedia.yaml",
    "ontology/patterns.rhetorical.yaml",
    "src/slopdet/__init__.py",
    "src/slopdet/ontology.py",
    "src/slopdet/weaklabel.py",
    "src/slopdet/construction.py",
    "src/slopdet/report.py",
    "src/slopdet/verify.py",
    "src/slopdet/calibrate.py",
    "src/slopdet/cli.py",
    "src/slopdet/teacher.py",
    "src/slopdet/student.py",
    "src/slopdet/heads.py",
    "src/slopdet/jlens.py",
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
            """# Is This AI Slop? (ITAIS) — Colab Run All

**Runtime → Change runtime type → T4 GPU** (CPU works for smoke). Then **Runtime → Run all**.

Smoke trains a calibrated `matches_ai_pile` head on weak labels. It does not claim authorship.
Gemma-3-4B residual distillation is optional (`FULL = True` + Hugging Face token with Gemma access).
If Gemma fails, Qwen2.5-0.5B is the fallback. If there is no GPU, that cell prints `skipped` and the demo still runs.
""",
        ),
        nb_cell(
            "code",
            """# Config
SMOKE = True          # False = more HC3 rows + try Gemma
FULL = False          # True requires HF_TOKEN with Gemma license accepted
MOUNT_DRIVE = True
N_DOCS = 400 if SMOKE else 4000

import os, sys
from pathlib import Path

try:
    from google.colab import drive, userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    drive = None
    userdata = None

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

if IN_COLAB:
    try:
        tok = userdata.get("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
            print("HF_TOKEN loaded from Colab secrets")
    except Exception:
        pass
if FULL:
    os.environ["FULL"] = "1"

import subprocess, sys
pkgs = ["pyyaml", "regex", "jsonschema", "scikit-learn", "datasets", "numpy>=1.26"]
if IN_COLAB:
    pkgs += ["transformers", "accelerate", "bitsandbytes", "safetensors"]
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
            """# Train smoke detector (seed + HC3 if downloadable)
from pathlib import Path
import sys
sys.path.insert(0, str(Path("src").resolve()))
sys.path.insert(0, str(Path("notebooks").resolve()))
from colab_pipeline import build_and_train, demo, try_gpu_distill

state = build_and_train(Path(".").resolve(), n_docs=N_DOCS)
demo(state)
""",
        ),
        nb_cell(
            "code",
            """# Optional GPU distill (skipped automatically on CPU / failed teacher load)
from pathlib import Path
try_gpu_distill(Path(".").resolve(), state["docs"], max_docs=32 if SMOKE else 256)
print("done")
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
