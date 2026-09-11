from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from app.datasets.acquisition import acquire_dataset


def test_archive_is_extracted_and_repeated_content_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "Training" / "glioma" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("L", (8, 8), color=100).save(image_path)
    archive = tmp_path / "dataset.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.write(image_path, "BrainTumor/Training/glioma/sample.png")

    raw_root = tmp_path / "raw"
    first = acquire_dataset(archive_path=archive, raw_root=raw_root)
    second = acquire_dataset(archive_path=archive, raw_root=raw_root)

    assert first["status"] == "COMPLETE"
    assert second["status"] == "SKIPPED"
    assert first["dataset_version_hash"] == second["dataset_version_hash"]
    assert (Path(first["root"]) / "Training/glioma/sample.png").is_file()


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", "unsafe")

    try:
        acquire_dataset(archive_path=archive, raw_root=tmp_path / "raw")
    except Exception as exc:  # noqa: BLE001
        assert "unsafe path" in str(exc).lower()
    else:
        raise AssertionError("unsafe archive was accepted")
