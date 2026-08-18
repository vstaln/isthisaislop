"""CLI: print checkable why-slop and why-human. Never a percentage-of-AI verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slopdet.explain import explain


def detect(text: str, ontology_dir: Path | None = None) -> dict:
    return explain(text, ontology_dir)


def _print_report(result: dict) -> None:
    print(f"lean: {result['lean']}")
    print()
    why_slop = result.get("why_slop") or []
    if why_slop:
        print("Why slop")
        for hit in why_slop:
            quote = hit.get("quote") or ""
            print(f"- {hit['id']}: {quote!r}" if quote else f"- {hit['id']}")
            note = hit.get("say") or hit.get("fix") or ""
            if note:
                print(f"  {note}")
    else:
        print(result.get("style_summary") or "Nothing matched.")
    print()
    why_human = result.get("why_human") or []
    print("Why human")
    if why_human:
        for hit in why_human:
            quote = hit.get("quote") or ""
            if quote:
                print(f"- {hit['id']}: {quote!r}")
            else:
                print(f"- {hit['id']}")
            print(f"  {hit.get('say') or hit.get('fix') or ''}")
    else:
        print("No human-style cues named.")
    print()
    print("Sentences")
    for sent in result.get("sentences") or []:
        print(f"- [{sent['lean']}] {sent['text']}")
    if result.get("resemblance"):
        print()
        print(result["resemblance"]["text"])


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
    _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
