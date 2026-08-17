"""CLI: print checkable hits. Never a percentage-of-AI verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slopdet.construction import construction_stats
from slopdet.ontology import default_ontology_dir, load_ontology
from slopdet.report import render_hits
from slopdet.weaklabel import label_text


def detect(text: str, ontology_dir: Path | None = None) -> dict:
    onto = load_ontology(ontology_dir or default_ontology_dir())
    hits = label_text(text, onto)
    result = render_hits(hits)
    result["construction"] = construction_stats(text)
    result["ontology_sha256"] = onto.sha256
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="itais")
    parser.add_argument("text", nargs="?", help="Text to scan. Defaults to stdin.")
    parser.add_argument("--ontology", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()
    result = detect(text, args.ontology)
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    print(result.get("style_summary") or "")
    for hit in result["hits"]:
        print(f"- {hit['id']}: {hit['quote']!r}")
        print(f"  fix: {hit['fix']}")
    if result.get("resemblance"):
        print(result["resemblance"]["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
