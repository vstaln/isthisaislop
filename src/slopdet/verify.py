"""Fail-closed artifact verification. Hash mismatch → empty hits, no regex fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class UnverifiedArtifact(Exception):
    def __init__(self, payload: dict[str, Any]):
        super().__init__(payload["status"])
        self.payload = payload


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(artifacts_dir: Path) -> dict[str, Any]:
    path = artifacts_dir / "MANIFEST.json"
    if not path.is_file():
        raise UnverifiedArtifact(unverified_payload("missing_manifest"))
    return json.loads(path.read_text(encoding="utf-8"))


def unverified_payload(reason: str = "unverified_artifact") -> dict[str, Any]:
    return {"status": "unverified_artifact", "reason": reason, "hits": [], "resemblance": None}


def verify_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    try:
        manifest = load_manifest(artifacts_dir)
    except UnverifiedArtifact:
        return unverified_payload("missing_manifest")
    files = manifest.get("files") or {}
    for rel, expected in files.items():
        path = artifacts_dir / rel
        if not path.is_file():
            return unverified_payload(f"missing:{rel}")
        actual = sha256_file(path)
        if actual != expected:
            return unverified_payload(f"mismatch:{rel}")
    return {"status": "ok", "manifest": manifest}
