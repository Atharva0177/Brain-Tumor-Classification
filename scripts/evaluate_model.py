#!/usr/bin/env python
"""Evaluate a checkpoint only on the untouched Testing partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.gate import evaluate_checkpoint, gradcam_sanity_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 5 held-out evaluation gate.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-gallery", type=int, default=None)
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(args.manifest, args.checkpoint, args.output_root, args.device, args.max_gallery)
    result["gradcam"] = gradcam_sanity_check(args.manifest, args.checkpoint, args.output_root, args.device)
    (args.output_root / "evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
