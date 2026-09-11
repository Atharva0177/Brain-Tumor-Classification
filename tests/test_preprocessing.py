from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.preprocessing.config import PreprocessingConfig
from app.preprocessing.pipeline import build_cache, write_preview_grid, write_split_artifact
from app.preprocessing.transforms import augment_image, crop_brain_region, deterministic_tensor

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")


def _dataset(root: Path, count: int = 10) -> Path:
    for split in ("Training", "Testing"):
        for class_index, class_name in enumerate(CLASSES):
            for index in range(count if split == "Training" else max(2, count // 2)):
                pixels = np.zeros((40, 50), dtype=np.uint8)
                pixels[5 + class_index : 30 + class_index, 8 + index : 30 + index] = 80 + class_index * 30
                path = root / split / class_name / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
    return root


def test_config_hash_is_stable_and_changes_with_parameters() -> None:
    first = PreprocessingConfig()
    second = PreprocessingConfig()

    assert first.hash() == second.hash()
    assert first.hash() != PreprocessingConfig(seed=99).hash()


def test_transform_returns_normalized_three_channel_tensor_and_fallback() -> None:
    config = PreprocessingConfig(image_size=32)
    tensor = deterministic_tensor(Image.new("L", (20, 20), color=0), config)
    cropped, fallback = crop_brain_region(Image.new("RGB", (20, 20), color="black"))

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert cropped.size == (20, 20)
    assert fallback is True


def test_cache_split_and_cache_hit_are_reproducible(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    config = PreprocessingConfig(image_size=32)

    first = build_cache(root, tmp_path / "processed", config)
    second = build_cache(root, tmp_path / "processed", config)
    split_path = write_split_artifact(first, tmp_path / "artifacts")
    preview_path = write_preview_grid(root, first, tmp_path / "artifacts", count=4)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(first["records"]) == 60
    assert {record["partition"] for record in first["records"]} == {"train", "validation", "testing"}
    assert split_path.is_file()
    assert preview_path.is_file()
    assert all(
        Path(tmp_path / "processed" / config.hash() / record["processed_path"]).is_file() for record in first["records"]
    )


def test_config_change_creates_new_cache(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    first = build_cache(root, tmp_path / "processed", PreprocessingConfig(image_size=32))
    second = build_cache(root, tmp_path / "processed", PreprocessingConfig(image_size=40))

    assert first["config_hash"] != second["config_hash"]
    assert first["cache_hit"] is False
    assert second["cache_hit"] is False


def test_augmentation_is_seeded_and_presets_are_not_cached() -> None:
    config = PreprocessingConfig()
    image = Image.new("RGB", (32, 32), color="gray")
    first = augment_image(image, config.augmentation_presets["medium"], seed=7)
    second = augment_image(image, config.augmentation_presets["medium"], seed=7)

    assert list(first.getdata()) == list(second.getdata())
    assert "augmentation" not in json_config(config)


def json_config(config: PreprocessingConfig) -> dict:
    return config.as_dict()
