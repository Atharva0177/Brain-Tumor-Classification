from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from app.config import settings
from app.datasets.hashing import dataset_version_hash


class DatasetAcquisitionError(RuntimeError):
    pass


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise DatasetAcquisitionError(f"Archive contains unsafe path: {member.filename}")
        zipped.extractall(destination)


def _download_kaggle_archive(dataset: str, destination: Path) -> Path:
    if not settings.kaggle_username or not settings.kaggle_key:
        raise DatasetAcquisitionError("KAGGLE_USERNAME and KAGGLE_KEY are required for Kaggle acquisition")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=str(destination), unzip=False, quiet=False)
    except Exception as exc:  # noqa: BLE001
        raise DatasetAcquisitionError(f"Kaggle download failed for {dataset}: {exc}") from exc

    archives = sorted(destination.glob("*.zip"))
    if len(archives) != 1:
        raise DatasetAcquisitionError(f"Expected one Kaggle archive in {destination}, found {len(archives)}")
    return archives[0]


def acquire_dataset(
    dataset: str | None = None,
    archive_path: Path | None = None,
    raw_root: Path | None = None,
    min_archive_bytes: int | None = None,
    max_archive_bytes: int | None = None,
) -> dict:
    """Download or import a dataset, extract it, and create its immutable manifest."""
    raw_root = raw_root or Path("data/raw")
    staging = raw_root / ".downloads"
    staging.mkdir(parents=True, exist_ok=True)
    archive = (
        Path(archive_path) if archive_path else _download_kaggle_archive(dataset or settings.kaggle_dataset, staging)
    )
    if not archive.is_file():
        raise DatasetAcquisitionError(f"Dataset archive does not exist: {archive}")

    size = archive.stat().st_size
    if min_archive_bytes is not None and size < min_archive_bytes:
        raise DatasetAcquisitionError(f"Archive is smaller than expected: {size} < {min_archive_bytes}")
    if max_archive_bytes is not None and size > max_archive_bytes:
        raise DatasetAcquisitionError(f"Archive is larger than expected: {size} > {max_archive_bytes}")

    extraction = staging / archive.stem
    if extraction.exists():
        shutil.rmtree(extraction)
    extraction.mkdir()
    try:
        _safe_extract(archive, extraction)
    except zipfile.BadZipFile as exc:
        raise DatasetAcquisitionError(f"Downloaded archive is not a valid ZIP: {archive}") from exc

    # Unpack single-directory archives so the expected Training/Testing root is stable.
    children = [path for path in extraction.iterdir() if not path.name.startswith(".")]
    content_root = children[0] if len(children) == 1 and children[0].is_dir() else extraction
    version_hash, manifest = dataset_version_hash(content_root)
    version_root = raw_root / version_hash
    if version_root.exists():
        return {
            "status": "SKIPPED",
            "dataset_version_hash": version_hash,
            "root": str(version_root),
            "manifest": manifest,
        }

    shutil.copytree(content_root, version_root)
    manifest_path = version_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "status": "COMPLETE",
        "dataset_version_hash": version_hash,
        "root": str(version_root),
        "archive": str(archive),
        "archive_size": size,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }
