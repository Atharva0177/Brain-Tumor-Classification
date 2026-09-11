from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class AugmentationPreset:
    rotation_degrees: float
    horizontal_flip_probability: float
    brightness_jitter: float
    contrast_jitter: float


@dataclass(frozen=True)
class PreprocessingConfig:
    version: str = "1"
    image_size: int = 224
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    validation_fraction: float = 0.15
    seed: int = 42
    crop_enabled: bool = True
    augmentation_presets: dict[str, AugmentationPreset] = field(
        default_factory=lambda: {
            "light": AugmentationPreset(5.0, 0.25, 0.05, 0.05),
            "medium": AugmentationPreset(10.0, 0.5, 0.10, 0.10),
            "heavy": AugmentationPreset(15.0, 0.5, 0.10, 0.10),
        }
    )

    def as_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
