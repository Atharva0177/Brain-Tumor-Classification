#!/usr/bin/env python
"""Download, extract, verify, and stage a brain MRI dataset.

This script deliberately stops before preprocessing. A successful run leaves a
verified, immutable dataset under ``data/raw/<dataset_version_hash>`` that can
be consumed by the preprocessing phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python scripts/download_dataset.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.datasets.acquisition import DatasetAcquisitionError, acquire_dataset
from app.datasets.verification import DEFAULT_CLASSES, verify_dataset, write_verification_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download/extract a dataset, verify it, and prepare it for preprocessing."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset",
        default=None,
        help="Kaggle dataset slug (owner/dataset-name). Defaults to KAGGLE_DATASET or the application default.",
    )
    source.add_argument("--archive", type=Path, help="Use an existing local ZIP archive instead of Kaggle.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Versioned raw-data root.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/verification"),
        help="Directory for verification JSON and Markdown reports.",
    )
    parser.add_argument("--min-archive-mb", type=float, help="Reject archives smaller than this size.")
    parser.add_argument("--max-archive-mb", type=float, help="Reject archives larger than this size.")
    parser.add_argument(
        "--class-names",
        nargs=4,
        default=list(DEFAULT_CLASSES),
        metavar=("CLASS1", "CLASS2", "CLASS3", "CLASS4"),
        help="Canonical class directory names in Training and Testing.",
    )
    parser.add_argument("--train-count", type=int, default=1400, help="Expected images per class in Training.")
    parser.add_argument("--test-count", type=int, default=400, help="Expected images per class in Testing.")
    parser.add_argument(
        "--count-tolerance",
        type=int,
        default=0,
        help="Allowed per-class count deviation. Findings remain visible in the report.",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=4,
        help="Maximum DCT pHash Hamming distance for a hard near-duplicate finding.",
    )
    parser.add_argument(
        "--near-duplicate-mse",
        type=float,
        default=0.0005,
        help="Maximum normalized thumbnail MSE in addition to the pHash threshold.",
    )
    parser.add_argument(
        "--allow-near-duplicates",
        action="store_true",
        help="Keep near-duplicate findings in the report without blocking preprocessing readiness.",
    )
    return parser


def _bytes_from_mb(value: float | None) -> int | None:
    return None if value is None else int(value * 1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        acquisition = acquire_dataset(
            dataset=args.dataset,
            archive_path=args.archive,
            raw_root=args.raw_root,
            min_archive_bytes=_bytes_from_mb(args.min_archive_mb),
            max_archive_bytes=_bytes_from_mb(args.max_archive_mb),
        )
    except DatasetAcquisitionError as exc:
        print(f"Dataset acquisition failed: {exc}", file=sys.stderr)
        return 2

    dataset_root = Path(acquisition["root"])
    report = verify_dataset(
        dataset_root,
        dataset_version_hash=acquisition["dataset_version_hash"],
        class_names=tuple(args.class_names),
        expected_train_count=args.train_count,
        expected_test_count=args.test_count,
        count_tolerance=args.count_tolerance,
        near_duplicate_distance=args.near_duplicate_distance,
        near_duplicate_mse=args.near_duplicate_mse,
        allow_near_duplicates=args.allow_near_duplicates,
    )
    artifact_directory = args.artifact_root / acquisition["dataset_version_hash"]
    json_path, markdown_path = write_verification_artifacts(report, artifact_directory)
    summary = {
        "status": report["status"],
        "acquisition_status": acquisition["status"],
        "dataset_version_hash": acquisition["dataset_version_hash"],
        "dataset_root": str(dataset_root),
        "verification_json": str(json_path),
        "verification_markdown": str(markdown_path),
        "image_count": report["summary"]["image_count"],
        "corrupt_count": report["summary"]["corrupt_count"],
        "hard_failure_count": len(report["hard_failures"]),
    }
    print(json.dumps(summary, indent=2))

    if report["status"] != "COMPLETE":
        print(f"Dataset is not ready for preprocessing. See {markdown_path}.", file=sys.stderr)
        return 1
    print(f"Dataset is ready for preprocessing: {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
