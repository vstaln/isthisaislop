"""Load and freeze the slop pattern ontology.

Ids are append-only. Disable a pattern with enabled=false; do not delete it.
ONTOLOGY_SHA256 is sha256 of the three YAML files concatenated in this order:
patterns.core.yaml, patterns.wikipedia.yaml, patterns.rhetorical.yaml.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import regex as regex_mod
import yaml
from jsonschema import Draft202012Validator

YAML_NAMES = (
    "patterns.core.yaml",
    "patterns.wikipedia.yaml",
    "patterns.rhetorical.yaml",
)


def default_ontology_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "ontology"


class OntologyError(ValueError):
    """Invalid ontology data."""


@dataclass(frozen=True)
class Pattern:
    id: str
    lane: str
    unit: str
    detector: str
    pattern: str
    fix: str
    source: str
    license: str
    min_len_words: int
    paper: str | None
    enabled: bool
    compiled: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class Ontology:
    patterns: tuple[Pattern, ...]
    sha256: str
    by_id: dict[str, Pattern]

    @property
    def ONTOLOGY_SHA256(self) -> str:
        return self.sha256

    def enabled_patterns(self) -> tuple[Pattern, ...]:
        return tuple(p for p in self.patterns if p.enabled)


def _schema(ontology_dir: Path) -> dict[str, Any]:
    return json.loads((ontology_dir / "schema.json").read_text(encoding="utf-8"))


def _load_yaml_list(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise OntologyError(f"{path.name} must be a YAML list, got {type(data).__name__}")
    return data


def load_ontology(ontology_dir: Path | None = None) -> Ontology:
    ontology_dir = Path(ontology_dir) if ontology_dir else default_ontology_dir()
    validator = Draft202012Validator(_schema(ontology_dir))
    seen: dict[str, str] = {}
    patterns: list[Pattern] = []
    raw_parts: list[bytes] = []

    for name in YAML_NAMES:
        path = ontology_dir / name
        raw_parts.append(path.read_bytes())
        for i, entry in enumerate(_load_yaml_list(path)):
            errors = sorted(validator.iter_errors(entry), key=lambda e: list(e.path))
            if errors:
                raise OntologyError(f"{name}[{i}]: {errors[0].message}")
            pid = entry["id"]
            if pid in seen:
                raise OntologyError(f"duplicate id {pid!r} in {name} and {seen[pid]}")
            seen[pid] = name
            try:
                compiled = regex_mod.compile(entry["pattern"])
            except regex_mod.error as exc:
                raise OntologyError(f"{pid}: regex does not compile: {exc}") from exc
            patterns.append(
                Pattern(
                    id=pid,
                    lane=entry["lane"],
                    unit=entry["unit"],
                    detector=entry["detector"],
                    pattern=entry["pattern"],
                    fix=entry["fix"],
                    source=entry["source"],
                    license=entry["license"],
                    min_len_words=int(entry["min_len_words"]),
                    paper=entry["paper"],
                    enabled=bool(entry["enabled"]),
                    compiled=compiled,
                )
            )

    sha256 = hashlib.sha256(b"".join(raw_parts)).hexdigest()
    frozen = tuple(patterns)
    return Ontology(
        patterns=frozen,
        sha256=sha256,
        by_id={p.id: p for p in frozen},
    )


ONTOLOGY_SHA256: str | None = None


def ontology_sha256(ontology_dir: Path | None = None) -> str:
    global ONTOLOGY_SHA256
    if ONTOLOGY_SHA256 is None:
        ONTOLOGY_SHA256 = load_ontology(ontology_dir).sha256
    return ONTOLOGY_SHA256
