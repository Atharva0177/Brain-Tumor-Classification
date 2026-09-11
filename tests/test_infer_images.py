from pathlib import Path

from PIL import Image

from scripts.infer_images import image_paths, overlay_heatmap


def test_image_paths_supports_single_and_recursive_folder(tmp_path: Path) -> None:
    image = tmp_path / "one.png"
    nested = tmp_path / "nested" / "two.jpg"
    nested.parent.mkdir()
    Image.new("RGB", (8, 8)).save(image)
    Image.new("RGB", (8, 8)).save(nested)

    assert image_paths(image, None) == [image]
    assert image_paths(None, tmp_path) == sorted([image, nested])


def test_overlay_heatmap_preserves_original_size(tmp_path: Path) -> None:
    image = Image.new("RGB", (20, 15), color="white")
    heatmap = __import__("numpy").ones((8, 8), dtype=float)

    result = overlay_heatmap(image, heatmap)

    assert result.size == image.size
