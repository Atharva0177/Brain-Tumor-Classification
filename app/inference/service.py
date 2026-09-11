from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.evaluation.gate import GradCAM, target_layer
from app.observability import get_logger
from app.preprocessing.config import PreprocessingConfig
from app.preprocessing.transforms import deterministic_tensor
from app.training.data import CLASS_NAMES
from app.training.models import ModelSpec, build_model

logger = get_logger("inference")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    architecture: str
    model_name: str
    model_version: str
    training_run_id: str | None
    dataset_version_hash: str | None
    preprocessing_config_hash: str | None
    metrics: dict
    cam: GradCAM


class ProductionModelService:
    def __init__(self, poll_seconds: int = 30):
        self._lock = threading.RLock()
        self._loaded: LoadedModel | None = None
        self._last_error: str | None = None
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.reload_if_needed()
        self._thread = threading.Thread(target=self._poll, name="production-model-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            if self._loaded:
                self._loaded.cam.close()

    def _poll(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            self.reload_if_needed()

    def _registry_version(self):
        client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        versions = client.get_latest_versions(settings.production_model_name, stages=["Production"])
        return client, versions[0] if versions else None

    def reload_if_needed(self) -> bool:
        try:
            client, version = self._registry_version()
            if version is None:
                self._last_error = "No Production model is registered"
                return False
            with self._lock:
                if self._loaded and self._loaded.model_version == str(version.version):
                    return True
            tags = dict(version.tags)
            architecture = tags.get("architecture", "convnext_base")
            source_path = Path(version.source)
            checkpoint_files = list(source_path.glob("*.pt")) if source_path.is_dir() else []
            if checkpoint_files:
                checkpoint = torch.load(checkpoint_files[0], map_location="cpu", weights_only=False)
                architecture = checkpoint.get("architecture", architecture)
                model = build_model(ModelSpec(architecture, pretrained=False))
                model.load_state_dict(checkpoint["model"])
            else:
                model = mlflow.pytorch.load_model(version.source, dst_path=None)
            model.to("cpu")
            model.eval()
            cam = GradCAM(model, target_layer(model, architecture))
            metrics = {}
            if version.run_id:
                run = client.get_run(version.run_id)
                metrics = dict(run.data.metrics)
            loaded = LoadedModel(
                model,
                architecture,
                settings.production_model_name,
                str(version.version),
                version.run_id,
                tags.get("dataset_version_hash"),
                tags.get("preprocessing_config_hash"),
                metrics,
                cam,
            )
            with self._lock:
                old = self._loaded
                self._loaded = loaded
                self._last_error = None
            if old:
                old.cam.close()
            logger.info("production_model_loaded model=%s version=%s", settings.production_model_name, version.version)
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            logger.exception("production_model_reload_failed")
            return False

    def status(self) -> dict:
        with self._lock:
            loaded = self._loaded
            return {
                "loaded": loaded is not None,
                "model_name": loaded.model_name if loaded else None,
                "model_version": loaded.model_version if loaded else None,
                "architecture": loaded.architecture if loaded else None,
                "error": self._last_error,
            }

    def info(self) -> dict:
        with self._lock:
            if not self._loaded:
                raise RuntimeError(self._last_error or "No Production model loaded")
            loaded = self._loaded
            return {
                "model_name": loaded.model_name,
                "model_version": loaded.model_version,
                "architecture": loaded.architecture,
                "training_run_id": loaded.training_run_id,
                "dataset_version_hash": loaded.dataset_version_hash,
                "preprocessing_config_hash": loaded.preprocessing_config_hash,
                "evaluation_metrics": loaded.metrics,
            }

    def predict(self, image: Image.Image) -> dict:
        with self._lock:
            if not self._loaded:
                raise RuntimeError(self._last_error or "No Production model loaded")
            loaded = self._loaded
            config = PreprocessingConfig()
            tensor = deterministic_tensor(image, config)[None]
            with torch.no_grad():
                probabilities = torch.softmax(loaded.model(tensor), dim=1)[0]
            predicted_index = int(probabilities.argmax())
            cam_tensor = tensor.requires_grad_(True)
            heatmap = loaded.cam.generate(cam_tensor, predicted_index)[0].detach().cpu().numpy()
            overlay = _overlay(image, heatmap)
            return {
                "predicted_class": CLASS_NAMES[predicted_index],
                "confidence": float(probabilities[predicted_index]),
                "probabilities": {name: float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)},
                "gradcam_overlay_base64": base64.b64encode(overlay).decode("ascii"),
                "model": self.info(),
            }


def _overlay(image: Image.Image, heatmap: np.ndarray) -> bytes:
    original = image.convert("RGB")
    resized = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").resize(
        original.size, Image.Resampling.BILINEAR
    )
    values = np.asarray(resized, dtype=np.float32) / 255.0
    color = np.zeros((*values.shape, 3), dtype=np.uint8)
    color[..., 0] = np.uint8(255 * values)
    color[..., 1] = np.uint8(180 * (1 - np.abs(values - 0.5) * 2))
    color[..., 2] = np.uint8(255 * (1 - values))
    output = io.BytesIO()
    Image.blend(original, Image.fromarray(color, mode="RGB"), 0.42).save(output, format="PNG")
    return output.getvalue()
