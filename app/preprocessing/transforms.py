from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage

from app.preprocessing.config import AugmentationPreset, PreprocessingConfig


def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)[:256]
    total = gray.size
    cumulative_weight = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))
    denominator = cumulative_weight * (total - cumulative_weight)
    between = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    total_mean = cumulative_mean[-1]
    between[valid] = (total * cumulative_mean[valid] - total_mean * cumulative_weight[valid]) ** 2 / denominator[valid]
    return int(np.argmax(between))


def crop_brain_region(image: Image.Image) -> tuple[Image.Image, bool]:
    """Crop the largest foreground component; return fallback status for empty masks."""
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    mask = gray > _otsu_threshold(gray)
    height, width = mask.shape
    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count:
        component_sizes = np.bincount(labels.ravel())[1:]
        largest = int(np.argmax(component_sizes)) + 1
        rows, columns = np.nonzero(labels == largest)
        best = (
            int(component_sizes[largest - 1]),
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        )
    else:
        best = None
    if best is None or best[0] < max(16, int(width * height * 0.01)):
        return rgb, True
    _, left, top, right, bottom = best
    padding = max(2, int(0.03 * max(right - left, bottom - top)))
    return rgb.crop(
        (max(0, left - padding), max(0, top - padding), min(width, right + padding), min(height, bottom + padding))
    ), False


def deterministic_tensor(image: Image.Image, config: PreprocessingConfig) -> torch.Tensor:
    working = image.convert("RGB")
    if config.crop_enabled:
        working, _ = crop_brain_region(working)
    working = working.resize((config.image_size, config.image_size), Image.Resampling.BILINEAR)
    array = np.asarray(working, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    mean = torch.tensor(config.normalize_mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.normalize_std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def augment_image(image: Image.Image, preset: AugmentationPreset, seed: int | None = None) -> Image.Image:
    generator = random.Random(seed)
    result = image.convert("RGB")
    if preset.rotation_degrees:
        result = result.rotate(
            generator.uniform(-preset.rotation_degrees, preset.rotation_degrees), resample=Image.Resampling.BILINEAR
        )
    if generator.random() < preset.horizontal_flip_probability:
        result = ImageOps.mirror(result)
    brightness = 1 + generator.uniform(-preset.brightness_jitter, preset.brightness_jitter)
    contrast = 1 + generator.uniform(-preset.contrast_jitter, preset.contrast_jitter)
    return ImageEnhance.Contrast(ImageEnhance.Brightness(result).enhance(brightness)).enhance(contrast)
