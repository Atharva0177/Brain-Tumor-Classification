from pathlib import Path

from PIL import Image

from scripts.visualize_dataset import collect_records, main


def test_visualization_script_creates_charts_and_report(tmp_path: Path) -> None:
    for split in ("Training", "Testing"):
        for class_name in ("glioma", "meningioma", "notumor", "pituitary"):
            directory = tmp_path / "dataset" / split / class_name
            directory.mkdir(parents=True)
            Image.new("RGB", (32, 24), color=(100, 120, 140)).save(directory / "sample.png")

    assert len(collect_records(tmp_path / "dataset")) == 8
    assert main(["--dataset-root", str(tmp_path / "dataset"), "--output-root", str(tmp_path / "viz")]) == 0
    assert (tmp_path / "viz" / "README.md").is_file()
    assert (tmp_path / "viz" / "class_distribution.png").is_file()
    assert (tmp_path / "viz" / "data_flow.png").is_file()
    assert (tmp_path / "viz" / "summary.json").is_file()
