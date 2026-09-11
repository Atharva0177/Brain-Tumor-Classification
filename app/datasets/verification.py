from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from app.datasets.hashing import md5_file

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")


def _perceptual_hash(path: Path, size: int = 32, low_frequency_size: int = 8) -> np.ndarray:
    """Return a DCT pHash bit vector, excluding the image's average brightness."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    pixels -= pixels.mean()
    coordinates = np.arange(size, dtype=np.float32)
    basis = np.cos(np.pi * (2 * coordinates[:, None] + 1) * coordinates[None, :] / (2 * size))
    dct = basis @ pixels @ basis.T
    low_frequency = dct[:low_frequency_size, :low_frequency_size].flatten()
    median = np.median(low_frequency[1:])
    return low_frequency > median


def _normalized_thumbnail(path: Path, size: int = 32) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    minimum, maximum = float(pixels.min()), float(pixels.max())
    if maximum == minimum:
        return np.zeros_like(pixels)
    return (pixels - minimum) / (maximum - minimum)


def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def verify_dataset(
    root: Path,
    dataset_version_hash: str | None = None,
    class_names: tuple[str, ...] = DEFAULT_CLASSES,
    expected_train_count: int = 1400,
    expected_test_count: int = 400,
    count_tolerance: int = 0,
    near_duplicate_distance: int = 4,
    near_duplicate_mse: float = 0.0005,
    allow_near_duplicates: bool = False,
) -> dict:
    """Run hard and soft checks and return a JSON-serializable verification report."""
    root = Path(root)
    hard_failures: list[dict] = []
    soft_findings: list[dict] = []
    required_roots = {"Training": expected_train_count, "Testing": expected_test_count}
    files_by_split: dict[str, list[Path]] = {}
    dimensions = Counter()
    formats = Counter()
    sizes_by_class: dict[str, list[int]] = defaultdict(list)
    corrupt: list[str] = []

    for split, expected_count in required_roots.items():
        split_root = root / split
        if not split_root.is_dir():
            hard_failures.append({"check": "structure", "split": split, "detail": "missing split directory"})
            files_by_split[split] = []
            continue
        actual_classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        missing = sorted(set(class_names) - set(actual_classes))
        unexpected = sorted(set(actual_classes) - set(class_names))
        if missing or unexpected:
            hard_failures.append({"check": "structure", "split": split, "missing": missing, "unexpected": unexpected})
        files = _image_files(split_root)
        files_by_split[split] = files
        if abs(len(files) - expected_count * len(class_names)) > count_tolerance * len(class_names):
            hard_failures.append(
                {
                    "check": "class_counts",
                    "split": split,
                    "expected_total": expected_count * len(class_names),
                    "actual_total": len(files),
                }
            )
        for class_name in class_names:
            count = len(_image_files(split_root / class_name))
            if abs(count - expected_count) > count_tolerance:
                hard_failures.append(
                    {
                        "check": "class_counts",
                        "split": split,
                        "class": class_name,
                        "expected": expected_count,
                        "actual": count,
                    }
                )

    hashes_by_file: dict[str, str] = {}
    phashes: dict[str, np.ndarray] = {}
    thumbnails: dict[str, np.ndarray] = {}
    for _split, files in files_by_split.items():
        for path in files:
            relative = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
                    formats[image.format or "unknown"] += 1
                hashes_by_file[relative] = md5_file(path)
                phashes[relative] = _perceptual_hash(path)
                thumbnails[relative] = _normalized_thumbnail(path)
                sizes_by_class[path.parent.name].append(path.stat().st_size)
            except Exception as exc:  # noqa: BLE001
                corrupt.append(relative)
                hard_failures.append({"check": "image_decode", "file": relative, "detail": str(exc)})

    train_hashes = {digest: path for path, digest in hashes_by_file.items() if path.startswith("Training/")}
    for path, digest in hashes_by_file.items():
        if path.startswith("Testing/") and digest in train_hashes:
            hard_failures.append(
                {"check": "exact_duplicate", "training_file": train_hashes[digest], "testing_file": path, "md5": digest}
            )

    train_phashes = {path: value for path, value in phashes.items() if path.startswith("Training/")}
    for test_path, test_hash in ((path, value) for path, value in phashes.items() if path.startswith("Testing/")):
        for train_path, train_hash in train_phashes.items():
            distance = int(np.count_nonzero(test_hash != train_hash))
            mse = float(np.mean((thumbnails[test_path] - thumbnails[train_path]) ** 2))
            if distance <= near_duplicate_distance and mse <= near_duplicate_mse:
                finding = {
                    "check": "near_duplicate",
                    "training_file": train_path,
                    "testing_file": test_path,
                    "hamming_distance": distance,
                    "threshold": near_duplicate_distance,
                    "normalized_mse": round(mse, 8),
                    "mse_threshold": near_duplicate_mse,
                }
                (soft_findings if allow_near_duplicates else hard_failures).append(finding)

    for class_name, sizes in sizes_by_class.items():
        if sizes:
            soft_findings.append(
                {
                    "check": "file_size_distribution",
                    "class": class_name,
                    "min": min(sizes),
                    "max": max(sizes),
                    "mean": round(sum(sizes) / len(sizes), 2),
                    "count": len(sizes),
                }
            )
    if len(formats) > 1:
        soft_findings.append({"check": "format_distribution", "formats": dict(formats)})

    return {
        "dataset_version_hash": dataset_version_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFICATION_FAILED" if hard_failures else "COMPLETE",
        "hard_failures": hard_failures,
        "soft_findings": soft_findings,
        "summary": {
            "image_count": sum(len(files) for files in files_by_split.values()),
            "corrupt_count": len(corrupt),
            "dimensions": dict(dimensions),
            "formats": dict(formats),
        },
        "thresholds": {
            "near_duplicate_distance": near_duplicate_distance,
            "near_duplicate_mse": near_duplicate_mse,
            "count_tolerance": count_tolerance,
            "allow_near_duplicates": allow_near_duplicates,
        },
    }


def write_verification_artifacts(report: dict, destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "verification_report.json"
    markdown_path = destination / "verification_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"# Dataset Verification: {report.get('status')}",
        "",
        f"Dataset version: `{report.get('dataset_version_hash') or 'unknown'}`",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["summary"].items())
    lines.extend(["", "## Hard failures", ""])
    lines.extend(
        f"- `{item.get('check')}`: {json.dumps(item, sort_keys=True)}" for item in report["hard_failures"]
    ) or lines.append("- None")
    lines.extend(["", "## Soft findings", ""])
    lines.extend(
        f"- `{item.get('check')}`: {json.dumps(item, sort_keys=True)}" for item in report["soft_findings"]
    ) or lines.append("- None")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
