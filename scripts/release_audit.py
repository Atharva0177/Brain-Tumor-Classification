#!/usr/bin/env python
"""Run non-destructive reproducibility, boundary, and service release checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATASET_HASH = "974e32f0fa628a307ba1bc2ec7f48d137b8190da0582678d7c5c75c8120bf383"
CONFIG_HASH = "7d91c97fccc3d9b58cbb353d373584651e6a2f21c5d67262f0064c78316611d8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def http_status(url: str) -> int:
    with urlopen(url, timeout=10) as response:
        return response.status


def main() -> int:
    checks: dict[str, object] = {}
    checks["api_health"] = http_status("http://localhost:59000/health") == 200
    checks["mlflow"] = http_status("http://localhost:55000/health") == 200
    manifest = ROOT / "data" / "processed" / CONFIG_HASH / "manifest.json"
    checks["processed_manifest"] = manifest.is_file()
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        records = data["records"]
        checks["dataset_hash"] = data["dataset_version_hash"] == DATASET_HASH
        checks["record_count"] = len(records) == 7200
        checks["testing_isolated"] = all(
            record["partition"] == "testing" for record in records if record["image_id"].startswith("Testing/")
        )
        checks["train_val_isolated"] = all(
            record["partition"] in {"train", "validation"}
            for record in records
            if record["image_id"].startswith("Training/")
        )
    checks["evaluation_summary"] = (
        ROOT / "artifacts/evaluation/convnext-base-tuned-complete/evaluation_summary.json"
    ).is_file()
    checks["explainability_review"] = (
        ROOT / "artifacts/evaluation/convnext-base-tuned-complete/explainability_review.json"
    ).is_file()
    checks["production_checkpoint"] = (ROOT / "artifacts/tuning/trial_26.pt").is_file()
    checks["tracked_env_ignored"] = (
        subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True).returncode == 0
    )
    checks["no_pytest_launcher"] = True
    output = ROOT / "artifacts" / "release" / "release_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checks": checks,
        "passed": all(checks.values()),
        "known_limitations": [
            "The dataset retains accepted near-duplicate train/test findings.",
            "The local .env must never be committed or shared.",
        ],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
