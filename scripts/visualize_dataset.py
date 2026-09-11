#!/usr/bin/env python
"""Generate charts, sample grids, and a visual data-flow report for an MRI dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
BG = "#f7f5f1"
PANEL = "#ffffff"
TEXT = "#1c252b"
MUTED = "#68747b"
INK = "#1c252b"
ACCENT = "#176b78"
ACCENT_LIGHT = "#a8d7d5"
WARM = "#d8734f"
CLASS_COLORS = {"glioma": "#176b78", "meningioma": "#d8734f", "notumor": "#557c63", "pituitary": "#8b6a9e"}


def styled_figure(size: tuple[float, float] = (14, 7)):
    fig, ax = plt.subplots(figsize=size, facecolor=BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#d9dedc")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(axis="y", color="#dfe4e1", alpha=0.8, linewidth=0.7)
    return fig, ax


def finish(fig, path: Path) -> Path:
    fig.tight_layout(pad=1.4)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def collect_records(dataset_root: Path) -> list[dict]:
    records = []
    for split in ("Training", "Testing"):
        split_root = dataset_root / split
        if not split_root.is_dir():
            continue
        for path in sorted(p for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            with Image.open(path) as image:
                width, height, image_format = image.width, image.height, image.format or path.suffix.upper().lstrip(".")
            records.append(
                {
                    "path": path,
                    "split": split,
                    "class_name": path.parent.name,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "size_bytes": path.stat().st_size,
                }
            )
    return records


def save_count_chart(records: list[dict], output: Path) -> Path:
    counts = Counter((r["split"], r["class_name"]) for r in records)
    classes = sorted({r["class_name"] for r in records})
    splits = ["Training", "Testing"]
    values = [[counts[(split, class_name)] for class_name in classes] for split in splits]
    fig, ax = styled_figure((14, 7))
    matrix = np.asarray(values)
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_title("Dataset balance by split", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=18)
    ax.set_xlabel("Class")
    ax.set_ylabel("Partition")
    ax.set_xticks(range(len(classes)), [name.title() for name in classes])
    ax.set_yticks(range(len(splits)), splits)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column, row, f"{int(matrix[row, column]):,}", ha="center", va="center", color=TEXT, fontweight="bold"
            )
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03, label="Images")
    ax.text(
        0,
        1.07,
        "Counts are expected to be balanced across all four classes.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    return finish(fig, output / "class_distribution.png")


def save_dimension_chart(records: list[dict], output: Path) -> Path:
    dimensions = Counter((r["width"], r["height"]) for r in records)
    top = dimensions.most_common(20)
    labels = [f"{width}x{height}" for (width, height), _ in top]
    values = [count for _, count in top]
    fig, ax = styled_figure((14, 7))
    ax.barh(labels[::-1], values[::-1], color=ACCENT, alpha=0.88)
    ax.set_title("Most common image dimensions", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=18)
    ax.set_xlabel("Image count")
    ax.xaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    ax.tick_params(axis="x", rotation=65)
    return finish(fig, output / "image_dimensions.png")


def save_file_size_chart(records: list[dict], output: Path) -> Path:
    by_class = defaultdict(list)
    for record in records:
        by_class[record["class_name"]].append(record["size_bytes"] / 1024)
    fig, ax = styled_figure((14, 7))
    classes = sorted(by_class)
    violin = ax.violinplot([by_class[name] for name in classes], showmeans=True, showextrema=False)
    for body, name in zip(violin["bodies"], classes, strict=True):
        body.set_facecolor(CLASS_COLORS.get(name, ACCENT))
        body.set_edgecolor(TEXT)
        body.set_alpha(0.7)
    ax.set_title("File-size distribution by class", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=18)
    ax.set_xticks(range(1, len(classes) + 1), [name.title() for name in classes])
    ax.set_ylabel("File size (KiB)")
    ax.text(
        0,
        1.07,
        "Distributions show variation in source encoding and image quality.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    return finish(fig, output / "file_size_distribution.png")


def save_format_chart(records: list[dict], output: Path) -> Path:
    formats = Counter(record["format"] for record in records)
    fig, ax = styled_figure((9, 6))
    keys, values = zip(*formats.most_common(), strict=True)
    ax.bar(keys, values, color=ACCENT)
    ax.set_title("Image formats", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=18)
    ax.set_ylabel("Images")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", color=TEXT, fontsize=10)
    return finish(fig, output / "format_distribution.png")


def save_sample_grid(records: list[dict], output: Path, split: str, samples_per_class: int = 4) -> Path:
    selected = []
    for class_name in sorted({r["class_name"] for r in records}):
        class_records = [r for r in records if r["split"] == split and r["class_name"] == class_name]
        selected.extend(class_records[:samples_per_class])
    tile_width, tile_height, columns = 300, 255, samples_per_class
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "#f7f5f1")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(selected):
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 28, tile_height - 62))
            left = (index % columns) * tile_width + (tile_width - image.width) // 2
            top = (index // columns) * tile_height + 4
            canvas.paste(image, (left, top))
            x = (index % columns) * tile_width
            y = (index // columns) * tile_height
            draw.rectangle((x + 9, y + 8, x + tile_width - 9, y + tile_height - 42), outline="#d9dedc", width=2)
            draw.text((x + 12, y + tile_height - 28), record["class_name"].title(), fill=TEXT)
    path = output / f"samples_{split.lower()}.png"
    canvas.save(path)
    return path


def save_flow_chart(output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(18, 5), facecolor=BG)
    ax.axis("off")
    labels = [
        ("01", "Kaggle / ZIP"),
        ("02", "Versioned raw"),
        ("03", "Verify"),
        ("04", "Preprocess"),
        ("05", "Tune"),
        ("06", "Evaluate"),
        ("07", "Serve"),
    ]
    for index, label in enumerate(labels):
        x = index / (len(labels) - 1)
        number, label_text = label
        ax.text(
            x,
            0.5,
            f"{number}\n{label_text}",
            ha="center",
            va="center",
            fontsize=13,
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": PANEL,
                "edgecolor": ACCENT if index in (0, 6) else "#c7d0cc",
                "linewidth": 1.5,
            },
            transform=ax.transAxes,
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=((index + 0.5) / (len(labels) - 1), 0.5),
                xytext=((index + 0.5) / (len(labels) - 1) - 0.06, 0.5),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": WARM},
            )
    ax.set_title("Brain MRI classification workflow", color=TEXT, loc="left", fontsize=19, fontweight="bold", pad=20)
    ax.text(
        0,
        0.08,
        "Every handoff creates a traceable artifact or registry record.",
        color=MUTED,
        transform=ax.transAxes,
        fontsize=10,
    )
    path = output / "data_flow.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(records: list[dict], output: Path, artifacts: list[Path], dataset_root: Path) -> Path:
    split_counts = Counter(record["split"] for record in records)
    class_counts = Counter(record["class_name"] for record in records)
    dimensions = Counter((record["width"], record["height"]) for record in records)
    report = [
        "# Dataset Visualization Report",
        "",
        f"Dataset root: `{dataset_root}`",
        "",
        "## Overview",
        "",
        f"- Total images: {len(records)}",
        f"- Training images: {split_counts['Training']}",
        f"- Testing images: {split_counts['Testing']}",
        f"- Classes: {', '.join(sorted(class_counts))}",
        f"- Most common dimensions: {dimensions.most_common(1)[0][0] if dimensions else 'n/a'}",
        "",
        "## Class counts",
        "",
    ]
    report.extend(f"- {name}: {count}" for name, count in sorted(class_counts.items()))
    report.extend(["", "## Generated artifacts", ""])
    report.extend(f"- [{path.name}]({path.name})" for path in artifacts)
    path = output / "README.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate charts, sample images, and a data-flow report.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/visualization"))
    parser.add_argument("--samples-per-class", type=int, default=4)
    args = parser.parse_args(argv)
    records = collect_records(args.dataset_root)
    if not records:
        parser.error(f"No supported images found below {args.dataset_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    artifacts = [
        save_count_chart(records, args.output_root),
        save_dimension_chart(records, args.output_root),
        save_file_size_chart(records, args.output_root),
        save_format_chart(records, args.output_root),
        save_sample_grid(records, args.output_root, "Training", args.samples_per_class),
        save_sample_grid(records, args.output_root, "Testing", args.samples_per_class),
        save_flow_chart(args.output_root),
    ]
    report_path = write_report(records, args.output_root, artifacts, args.dataset_root)
    summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "image_count": len(records),
        "artifacts": [str(path) for path in artifacts],
        "report": str(report_path),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
