#!/usr/bin/env python3
"""Fetch the training corpora into data/raw/.

coai is fetched on demand by scripts/train_cpu_scorer.py; this pulls the three companion
piles: StoryScope (AI fiction), GutenbergFiction (human fiction), and the
blog authorship corpus (human blogs — the register where slop actually lives).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JOBS = {
    "storyscope": {
        "files": {
            "stories_dev.parquet": "stories_dev.parquet",
            "stories_val.parquet": "stories_val.parquet",
            "stories_test.parquet": "stories_test.parquet",
            "stories_train.parquet": "stories_train.parquet",
        },
        "url": "https://huggingface.co/datasets/jjrussell10/storyscope/resolve/main/{f}",
        "license": "MIT (repo), human text excluded (Books3 copyright)",
    },
    "gutenberg_fiction": {
        "files": {
            "train-00000-of-00004.parquet": "train-00000-of-00004.parquet",
            "train-00001-of-00004.parquet": "train-00001-of-00004.parquet",
            "train-00002-of-00004.parquet": "train-00002-of-00004.parquet",
            "train-00003-of-00004.parquet": "train-00003-of-00004.parquet",
        },
        "url": "https://huggingface.co/datasets/sanps/GutenbergFiction/resolve/main/data/{f}",
        "license": "public domain (Project Gutenberg)",
    },
    "blogs": {
        "files": {"blogs.zip": "blogs.zip"},
        "url": "https://huggingface.co/datasets/barilan/blog_authorship_corpus/resolve/main/data/{f}",
        "license": "research use (Schler et al. 2006)",
        "unzip": True,
    },
    "writingprompts": {
        "files": {
            "train-00000-of-00002-105e07cb0d199464.parquet": "train-0.parquet",
            "train-00001-of-00002-4fdb982c11056472.parquet": "train-1.parquet",
            "test-00000-of-00001-16503b0c26ed00c6.parquet": "test.parquet",
        },
        "url": "https://huggingface.co/datasets/euclaise/writingprompts/resolve/main/data/{f}",
        "license": "Reddit user content (The Pile, Apache-2.0 extraction)",
    },
    "scp": {
        "files": {
            "scp_tales_cleaned.jsonl": "scp_tales_cleaned.jsonl",
            "stories1_cleaned.jsonl": "stories1_cleaned.jsonl",
            "stories2_cleaned.jsonl": "stories2_cleaned.jsonl",
            "stories3_cleaned.jsonl": "stories3_cleaned.jsonl",
            "stories4_cleaned.jsonl": "stories4_cleaned.jsonl",
            "stories5_cleaned.jsonl": "stories5_cleaned.jsonl",
            "stories6_cleaned.jsonl": "stories6_cleaned.jsonl",
            "stories7_cleaned.jsonl": "stories7_cleaned.jsonl",
        },
        "url": "https://huggingface.co/datasets/recursal/SCP-RECURSAL/resolve/main/data/{f}",
        "license": "CC-BY-SA (SCP wiki)",
    },
}


def fetch(dest: Path, url: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"  fetching {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "itaisslop/0.1"})
    with urllib.request.urlopen(req) as r, dest.open("wb") as out:
        out.write(r.read())


def extract(dest: Path) -> Path:
    import zipfile

    out = dest.with_suffix("")  # blogs.zip -> blogs
    if out.is_dir():
        return out
    print(f"  unzipping {dest.name}", flush=True)
    with zipfile.ZipFile(dest) as zf:
        zf.extractall(out)
    return out


def main() -> None:
    which = sys.argv[1:] or list(JOBS)
    manifest: list[dict] = []
    for name in which:
        job = JOBS[name]
        print(name, flush=True)
        for remote, local in job["files"].items():
            dest = ROOT / "data" / "raw" / name / local
            fetch(dest, job["url"].format(f=remote))
            if job.get("unzip"):
                dest = extract(dest)
                for f in dest.rglob("*"):
                    if f.is_file():
                        manifest.append(
                            {
                                "dataset": name,
                                "file": str(f.relative_to(ROOT)),
                                "bytes": f.stat().st_size,
                                "license": job["license"],
                            }
                        )
                continue
            manifest.append(
                {
                    "dataset": name,
                    "file": str(dest.relative_to(ROOT)),
                    "bytes": dest.stat().st_size,
                    "license": job["license"],
                }
            )
    (ROOT / "data" / "raw" / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(f"manifest -> data/raw/manifest.json ({len(manifest)} files)")


if __name__ == "__main__":
    main()
