from pathlib import Path

import numpy as np
from PIL import Image

from app.datasets.hashing import dataset_version_hash
from app.datasets.verification import verify_dataset, write_verification_artifacts

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")


def _write_image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, columns = np.indices((32, 32))
    pixels = ((value + rows * 3 + columns * 5 + ((rows + columns) % 3) * 20) % 256).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image.save(path)


def _make_dataset(root: Path, count: int = 2) -> None:
    for split in ("Training", "Testing"):
        for class_index, class_name in enumerate(CLASSES):
            for image_index in range(count):
                split_offset = 0 if split == "Training" else 100
                _write_image(
                    root / split / class_name / f"{image_index}.png", 40 + split_offset + class_index * 35 + image_index
                )


def test_valid_fixture_passes_and_writes_reports(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_dataset(root)

    version_hash, manifest = dataset_version_hash(root)
    report = verify_dataset(root, version_hash, expected_train_count=2, expected_test_count=2)
    json_path, markdown_path = write_verification_artifacts(report, tmp_path / "artifacts")

    assert len(manifest) == 16
    assert report["status"] == "COMPLETE"
    assert report["hard_failures"] == []
    assert json_path.is_file()
    assert markdown_path.is_file()


def test_corrupt_image_is_reported_as_hard_failure(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_dataset(root)
    corrupt = root / "Testing" / "glioma" / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    report = verify_dataset(root, expected_train_count=2, expected_test_count=2, count_tolerance=1)

    assert report["status"] == "VERIFICATION_FAILED"
    assert any(
        item["check"] == "image_decode" and item["file"].endswith("corrupt.png") for item in report["hard_failures"]
    )


def test_exact_and_near_duplicates_are_reported(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_dataset(root)
    source = root / "Training" / "glioma" / "0.png"
    duplicate = root / "Testing" / "meningioma" / "duplicate.png"
    duplicate.write_bytes(source.read_bytes())
    near_duplicate = root / "Testing" / "pituitary" / "near.png"
    image = np.asarray(Image.open(source).convert("L"), dtype=np.int16)
    Image.fromarray(np.clip(image + 1, 0, 255).astype(np.uint8), mode="L").save(near_duplicate)

    report = verify_dataset(
        root,
        expected_train_count=2,
        expected_test_count=2,
        count_tolerance=1,
        near_duplicate_distance=4,
        near_duplicate_mse=0.0005,
    )
    checks = {item["check"] for item in report["hard_failures"]}

    assert "exact_duplicate" in checks
    assert "near_duplicate" in checks


def test_near_duplicates_can_be_retained_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _make_dataset(root)
    source = root / "Training" / "glioma" / "0.png"
    duplicate = root / "Testing" / "glioma" / "near.png"
    image = np.asarray(Image.open(source).convert("L"), dtype=np.int16)
    Image.fromarray(np.clip(image + 1, 0, 255).astype(np.uint8), mode="L").save(duplicate)

    report = verify_dataset(
        root,
        expected_train_count=2,
        expected_test_count=2,
        count_tolerance=1,
        allow_near_duplicates=True,
    )

    assert report["status"] == "COMPLETE"
    assert any(item["check"] == "near_duplicate" for item in report["soft_findings"])
