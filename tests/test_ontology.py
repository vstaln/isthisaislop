"""Phase 0: ontology freeze tests.

Written before ontology.py existed. Failures must be missing features, not typos.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_DIR = ROOT / "ontology"
SCHEMA_PATH = ONTOLOGY_DIR / "schema.json"
YAML_FILES = (
    ONTOLOGY_DIR / "patterns.core.yaml",
    ONTOLOGY_DIR / "patterns.wikipedia.yaml",
    ONTOLOGY_DIR / "patterns.rhetorical.yaml",
    ONTOLOGY_DIR / "patterns.slop.yaml",
)
REQUIRED_KEYS = {
    "id",
    "lane",
    "unit",
    "detector",
    "pattern",
    "fix",
    "source",
    "license",
    "min_len_words",
    "paper",
    "enabled",
}
LANES = {"style", "rhetorical", "construction"}
UNITS = {"span", "sentence", "paragraph", "piece"}
DETECTORS = {"regex", "heuristic", "model_only"}
LICENSES = {"MIT-compatible", "CC-BY-SA-4.0", "Apache-2.0"}
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _load_entries(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), f"{path.name} must be a YAML list"
    return data


def test_schema_file_exists_and_is_json_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert schema["type"] == "object"
    required = set(schema["required"])
    assert REQUIRED_KEYS <= required


def test_yaml_files_exist() -> None:
    for path in YAML_FILES:
        assert path.is_file(), f"missing {path}"


def test_every_entry_matches_schema_keys() -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for path in YAML_FILES:
        for i, entry in enumerate(_load_entries(path)):
            errors = sorted(validator.iter_errors(entry), key=lambda e: e.path)
            assert not errors, f"{path.name}[{i}] {entry.get('id')}: {errors[0].message}"
            extra = set(entry) - set(schema["properties"])
            assert not extra, f"{path.name}[{i}] extra keys {extra}"


def test_no_duplicate_ids() -> None:
    seen: dict[str, str] = {}
    for path in YAML_FILES:
        for entry in _load_entries(path):
            pid = entry["id"]
            assert ID_RE.match(pid), f"bad id {pid!r}"
            assert pid not in seen, f"duplicate id {pid} in {path.name} and {seen[pid]}"
            seen[pid] = path.name
    assert len(seen) >= 70, f"ontology-v1 needs ≥70 ids, got {len(seen)}"


def test_every_regex_compiles() -> None:
    import regex as regex_mod

    for path in YAML_FILES:
        for entry in _load_entries(path):
            try:
                regex_mod.compile(entry["pattern"])
            except Exception as exc:  # noqa: BLE001 — we want the pattern id in the message
                pytest.fail(f"{entry['id']}: regex does not compile: {exc}")


def test_every_entry_has_nonempty_fix() -> None:
    for path in YAML_FILES:
        for entry in _load_entries(path):
            fix = entry["fix"]
            assert isinstance(fix, str) and fix.strip(), f"{entry['id']} missing fix"
            assert len(fix.strip()) >= 8, f"{entry['id']} fix too short"


def test_licenses_declared() -> None:
    for path in YAML_FILES:
        for entry in _load_entries(path):
            assert entry["license"] in LICENSES, f"{entry['id']} bad license"
    wiki = _load_entries(ONTOLOGY_DIR / "patterns.wikipedia.yaml")
    assert wiki, "wikipedia file must not be empty"
    assert all(e["license"] == "CC-BY-SA-4.0" for e in wiki)
    core = _load_entries(ONTOLOGY_DIR / "patterns.core.yaml")
    rhet = _load_entries(ONTOLOGY_DIR / "patterns.rhetorical.yaml")
    assert all(e["license"] == "MIT-compatible" for e in core + rhet)


def test_wikipedia_file_carries_sharealike_header() -> None:
    text = (ONTOLOGY_DIR / "patterns.wikipedia.yaml").read_text(encoding="utf-8")
    assert "CC BY-SA 4.0" in text or "CC-BY-SA-4.0" in text
    assert "Signs of AI writing" in text
    assert "creativecommons.org/licenses/by-sa/4.0" in text


def test_load_ontology_rejects_duplicates_and_emits_sha256() -> None:
    from slopdet.ontology import load_ontology

    onto = load_ontology(ONTOLOGY_DIR)
    assert onto.sha256 == onto.ONTOLOGY_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", onto.sha256)
    raw = b"".join(p.read_bytes() for p in YAML_FILES)
    assert onto.sha256 == hashlib.sha256(raw).hexdigest()
    by_id = {p.id: p for p in onto.patterns}
    assert len(by_id) == len(onto.patterns)
    enabled = [p for p in onto.patterns if p.enabled]
    assert enabled
    assert all(p.compiled is not None for p in onto.patterns)


def test_disabled_ids_are_kept_not_dropped() -> None:
    from slopdet.ontology import load_ontology

    onto = load_ontology(ONTOLOGY_DIR)
    # Rollback contract: disabling an id must not change the id set.
    ids = {p.id for p in onto.patterns}
    disabled = {p.id for p in onto.patterns if not p.enabled}
    assert disabled <= ids
    assert onto.by_id.keys() == ids
