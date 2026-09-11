#!/usr/bin/env python
"""Run or resume one Optuna study per architecture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training.models import ARCHITECTURES
from app.tuning.studies import run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune one architecture with a resumable Optuna study.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument(
        "--storage", default=None, help="Optuna storage URL. Defaults to the dedicated Postgres schema."
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/tuning"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)
    summary = run_study(
        args.architecture,
        args.manifest,
        args.checkpoint_root,
        args.trials,
        args.storage,
        args.device,
        not args.no_pretrained,
        args.mlflow_tracking_uri,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
