from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

from app.datasets.hashing import dataset_version_hash
from app.preprocessing.config import PreprocessingConfig
from app.preprocessing.transforms import deterministic_tensor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_records(dataset_root: Path) -> list[dict[str, str]]:
    records = []
    for split in ("Training", "Testing"):
        split_root = dataset_root / split
        for path in sorted(p for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            records.append(
                {
                    "image_id": path.relative_to(dataset_root).as_posix(),
                    "source_path": str(path),
                    "split": split,
                    "class_name": path.parent.name,
                }
            )
    return records


def split_records(records: list[dict[str, str]], config: PreprocessingConfig) -> list[dict[str, str]]:
    training = [record for record in records if record["split"] == "Training"]
    testing = [record for record in records if record["split"] == "Testing"]
    ids = [record["image_id"] for record in training]
    labels = [record["class_name"] for record in training]
    train_ids, validation_ids = train_test_split(
        ids, test_size=config.validation_fraction, random_state=config.seed, stratify=labels
    )
    assignments = {image_id: "train" for image_id in train_ids} | {
        image_id: "validation" for image_id in validation_ids
    }
    return [{**record, "partition": assignments.get(record["image_id"], "testing")} for record in training + testing]


def _atomic_replace(source: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.old")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    source.rename(destination)
    if backup.exists():
        shutil.rmtree(backup)


def build_cache(dataset_root: Path, output_root: Path, config: PreprocessingConfig) -> dict:
    dataset_hash, _ = dataset_version_hash(dataset_root)
    config_hash = config.hash()
    cache_root = output_root / config_hash
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_version_hash") == dataset_hash and manifest.get("config_hash") == config_hash:
            return {**manifest, "cache_hit": True}

    temporary = output_root / f".{config_hash}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    tensors_root = temporary / "tensors"
    tensors_root.mkdir(parents=True)
    records = image_records(dataset_root)
    split_assignment = split_records(records, config)
    processed = []
    for record in split_assignment:
        relative_output = Path(record["image_id"]).with_suffix(".pt")
        output_path = tensors_root / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(record["source_path"]) as image:
            tensor = deterministic_tensor(image, config)
        torch.save(tensor, output_path)
        processed.append({**record, "processed_path": str(Path("tensors") / relative_output).replace("\\", "/")})
    manifest = {
        "dataset_version_hash": dataset_hash,
        "config_hash": config_hash,
        "config": config.as_dict(),
        "records": processed,
        "cache_hit": False,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _atomic_replace(temporary, cache_root)
    return manifest


def write_split_artifact(manifest: dict, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"split_{manifest['config_hash']}.json"
    path.write_text(
        json.dumps(
            {
                "dataset_version_hash": manifest["dataset_version_hash"],
                "config_hash": manifest["config_hash"],
                "records": [
                    {"image_id": r["image_id"], "class_name": r["class_name"], "partition": r["partition"]}
                    for r in manifest["records"]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_preview_grid(dataset_root: Path, manifest: dict, destination: Path, count: int = 16) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    selected = manifest["records"][:count]
    tile_size, columns = 160, 4
    grid = Image.new("RGB", (columns * tile_size, ((len(selected) + columns - 1) // columns) * tile_size), "white")
    draw = ImageDraw.Draw(grid)
    for index, record in enumerate(selected):
        with Image.open(record["source_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_size - 8, tile_size - 28))
            left = (index % columns) * tile_size + (tile_size - image.width) // 2
            top = (index // columns) * tile_size + 4
            grid.paste(image, (left, top))
            draw.text(
                ((index % columns) * tile_size + 4, (index // columns) * tile_size + tile_size - 20),
                record["image_id"][-28:],
                fill="black",
            )
    path = destination / f"preprocessing_preview_{manifest['config_hash']}.png"
    grid.save(path)
    return path
