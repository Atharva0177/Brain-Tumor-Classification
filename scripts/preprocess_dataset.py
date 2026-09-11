#!/usr/bin/env python
"""Build or reuse the hashed preprocessing cache and split artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.preprocessing.config import PreprocessingConfig
from app.preprocessing.pipeline import build_cache, write_preview_grid, write_split_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a verified dataset for model training.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/preprocessing"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-crop", action="store_true")
    args = parser.parse_args(argv)
    config = PreprocessingConfig(
        image_size=args.image_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        crop_enabled=not args.no_crop,
    )
    manifest = build_cache(args.dataset_root, args.output_root, config)
    split_path = write_split_artifact(manifest, args.artifact_root)
    preview_path = write_preview_grid(args.dataset_root, manifest, args.artifact_root)
    partitions = {}
    for record in manifest["records"]:
        partitions[record["partition"]] = partitions.get(record["partition"], 0) + 1
    print(
        json.dumps(
            {
                "cache_hit": manifest["cache_hit"],
                "dataset_version_hash": manifest["dataset_version_hash"],
                "config_hash": manifest["config_hash"],
                "record_count": len(manifest["records"]),
                "partitions": partitions,
                "cache_root": str(args.output_root / manifest["config_hash"]),
                "split_artifact": str(split_path),
                "preview_artifact": str(preview_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
