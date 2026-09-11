#!/usr/bin/env python
"""Run one reproducible baseline training job for a prepared cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
from torch.utils.data import DataLoader

from app.training.data import CachedTensorDataset
from app.training.models import ARCHITECTURES, ModelSpec, build_model
from app.training.trainer import TrainingConfig, train_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one baseline architecture on the cached train/validation split."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--fine-tune-epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args(argv)
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_dataset = CachedTensorDataset(args.manifest, "train")
    validation_dataset = CachedTensorDataset(args.manifest, "validation")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(ModelSpec(args.architecture, pretrained=not args.no_pretrained))
    config = TrainingConfig(
        architecture=args.architecture,
        dataset_version_hash=manifest["dataset_version_hash"],
        preprocessing_config_hash=manifest["config_hash"],
        split_id=manifest["config_hash"],
        head_warmup_epochs=args.head_warmup_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        device=args.device,
    )
    summary = train_model(model, train_loader, validation_loader, config, args.checkpoint_root)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
