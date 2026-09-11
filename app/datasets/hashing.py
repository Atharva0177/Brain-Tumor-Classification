from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_manifest(root: Path) -> list[dict[str, int | str]]:
    """Return a deterministic manifest, excluding transient dot-files."""
    files = []
    for path in sorted(
        p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".") and p.name != "manifest.json"
    ):
        files.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size})
    return files


def dataset_version_hash(root: Path) -> tuple[str, list[dict[str, int | str]]]:
    manifest = file_manifest(root)
    payload = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), manifest


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
