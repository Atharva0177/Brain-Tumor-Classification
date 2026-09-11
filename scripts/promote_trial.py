#!/usr/bin/env python
"""Register a completed tuning checkpoint in MLflow and move it to Staging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tuning.studies import promote_checkpoint_to_staging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register and stage a completed Optuna trial checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tracking-uri", default="http://localhost:55000")
    args = parser.parse_args(argv)
    result = promote_checkpoint_to_staging(
        args.checkpoint, args.model_name, json.loads(args.summary.read_text(encoding="utf-8")), args.tracking_uri
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
