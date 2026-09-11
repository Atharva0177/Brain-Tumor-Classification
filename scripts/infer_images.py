#!/usr/bin/env python
"""Run image or folder inference and save Grad-CAM-highlighted images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.gate import GradCAM, load_checkpoint, target_layer
from app.preprocessing.config import PreprocessingConfig
from app.preprocessing.transforms import deterministic_tensor
from app.training.data import CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(image: Path | None, folder: Path | None) -> list[Path]:
    if image:
        return [image] if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    original = image.convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").resize(
        original.size, Image.Resampling.BILINEAR
    )
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    # Simple blue-to-red heatmap without requiring OpenCV.
    color = np.zeros((*heatmap_array.shape, 3), dtype=np.uint8)
    color[..., 0] = np.uint8(255 * heatmap_array)
    color[..., 1] = np.uint8(180 * (1 - np.abs(heatmap_array - 0.5) * 2))
    color[..., 2] = np.uint8(255 * (1 - heatmap_array))
    colored = Image.fromarray(color, mode="RGB")
    return Image.blend(original, colored, alpha)


def run_inference(
    checkpoint: Path, inputs: list[Path], output_root: Path, device: str, crop_enabled: bool
) -> list[dict]:
    resolved_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    model, architecture, checkpoint_data = load_checkpoint(checkpoint, resolved_device)
    cam = GradCAM(model, target_layer(model, architecture))
    config = PreprocessingConfig(crop_enabled=crop_enabled)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for path in inputs:
            try:
                with Image.open(path) as source:
                    original = source.convert("RGB")
                tensor = deterministic_tensor(original, config)[None].to(resolved_device)
                with torch.no_grad():
                    probabilities = torch.softmax(model(tensor), dim=1)[0]
                predicted_index = int(probabilities.argmax().item())
                heatmap = cam.generate(tensor, predicted_index)[0].detach().cpu().numpy()
                highlighted = overlay_heatmap(original, heatmap)
                output_path = output_root / f"{path.stem}__{CLASS_NAMES[predicted_index]}__highlighted.png"
                highlighted.save(output_path)
                result = {
                    "input": str(path),
                    "output": str(output_path),
                    "architecture": architecture,
                    "predicted_class": CLASS_NAMES[predicted_index],
                    "confidence": float(probabilities[predicted_index]),
                    "probabilities": {name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)},
                    "status": "ok",
                }
            except Exception as exc:  # noqa: BLE001
                result = {"input": str(path), "status": "error", "error": str(exc)}
            results.append(result)
    finally:
        cam.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify one image or a folder and save Grad-CAM-highlighted results."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="One image to classify.")
    source.add_argument("--folder", type=Path, help="Folder of images. Subfolders are scanned recursively.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PyTorch checkpoint, e.g. convnext_base_best.pt")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/inference"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-crop", action="store_true", help="Disable the brain-region crop used during preprocessing."
    )
    args = parser.parse_args(argv)
    inputs = image_paths(args.image, args.folder)
    if not inputs:
        parser.error("No supported images found")
    results = run_inference(args.checkpoint, inputs, args.output_root, args.device, not args.no_crop)
    successful = [result for result in results if result["status"] == "ok"]
    with (args.output_root / "predictions.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    with (args.output_root / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["input", "output", "architecture", "predicted_class", "confidence", "status", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})
    print(
        json.dumps(
            {
                "processed": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "output_root": str(args.output_root),
                "predictions_json": str(args.output_root / "predictions.json"),
                "predictions_csv": str(args.output_root / "predictions.csv"),
            },
            indent=2,
        )
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
