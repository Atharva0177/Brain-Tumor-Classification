from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from app.training.data import CLASS_NAMES, CachedTensorDataset
from app.training.metrics import classification_metrics
from app.training.models import ModelSpec, build_model


def target_layer(model, architecture: str):
    if architecture == "resnet50":
        return model.layer4[-1].conv3
    if architecture in ("convnext_tiny", "convnext_base"):
        return model.features[-1][-1]
    if architecture == "efficientnet_b0":
        return model.features[-1][0]
    raise ValueError(f"Unsupported architecture {architecture!r}")


def target_layer_path(architecture: str) -> str:
    return {
        "resnet50": "layer4[-1].conv3",
        "convnext_tiny": "features[-1][-1]",
        "convnext_base": "features[-1][-1]",
        "efficientnet_b0": "features[-1][0]",
    }[architecture]


def validate_gradcam_target(
    model, architecture: str, input_shape: tuple[int, int, int, int] = (1, 3, 224, 224)
) -> dict:
    layer = target_layer(model, architecture)
    captured = {}

    def capture(_module, _inputs, output):
        captured["shape"] = tuple(output.shape)

    handle = layer.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(torch.zeros(input_shape, device=next(model.parameters()).device))
    finally:
        handle.remove()
    shape = captured.get("shape")
    if not shape or len(shape) != 4:
        raise ValueError(f"Grad-CAM target for {architecture} did not produce a 4D activation: {shape}")
    return {"architecture": architecture, "target_layer": target_layer_path(architecture), "activation_shape": shape}


def write_explainability_review_metadata(
    output_root: Path,
    model_name: str,
    model_version: str,
    architecture: str,
    checkpoint_path: Path,
    dataset_version_hash: str,
    preprocessing_config_hash: str,
    gradcam_result: dict,
    reviewer_status: str = "PENDING",
    review_notes: str = "",
) -> Path:
    """Write immutable explainability metadata once per model/evaluation artifact."""
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "explainability_review.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("model_version") != model_version or existing.get("checkpoint_sha256") != _file_sha256(
            checkpoint_path
        ):
            raise FileExistsError(f"Explainability metadata already exists for another model at {path}")
        return path
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "dataset_version_hash": dataset_version_hash,
        "preprocessing_config_hash": preprocessing_config_hash,
        "gradcam": gradcam_result,
        "reviewer_status": reviewer_status,
        "review_notes": review_notes,
        "immutable": True,
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = layer.register_forward_hook(self._forward)
        self.backward_handle = layer.register_full_backward_hook(self._backward)

    def _forward(self, module, inputs, output):
        self.activations = output.detach()

    def _backward(self, module, grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0].detach()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, inputs: torch.Tensor, class_index: int) -> torch.Tensor:
        self.gradients = None
        self.activations = None
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        logits[:, class_index].sum().backward()
        weights = self.gradients.mean(dim=tuple(range(2, self.gradients.ndim)), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()
        cam = torch.nn.functional.interpolate(
            cam[:, None], size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        cam_min = cam.flatten(1).min(1)[0][:, None, None]
        cam_max = cam.flatten(1).max(1)[0][:, None, None]
        return (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)


def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture") or checkpoint["config"]["architecture"]
    model = build_model(ModelSpec(architecture, pretrained=False))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, architecture, checkpoint


def evaluate_checkpoint(
    manifest_path: Path, checkpoint_path: Path, output_root: Path, device: str = "cuda", max_gallery: int | None = None
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    model, architecture, checkpoint = load_checkpoint(checkpoint_path, resolved_device)
    dataset = CachedTensorDataset(manifest_path, "testing")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    predictions, targets, probabilities = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(resolved_device))
            probabilities.extend(torch.softmax(logits, 1).cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    metrics = classification_metrics(predictions, targets)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for actual, predicted in zip(targets, predictions, strict=True):
        confusion[actual, predicted] += 1
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(
        json.dumps(
            {
                "architecture": architecture,
                "checkpoint": str(checkpoint_path),
                "dataset_version_hash": manifest["dataset_version_hash"],
                "preprocessing_config_hash": manifest["config_hash"],
                "device": resolved_device,
                "metrics": metrics,
                "confusion_matrix": confusion.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="Actual",
        title="Held-out Testing Confusion Matrix",
    )
    for row in range(4):
        for column in range(4):
            ax.text(column, row, confusion[row, column], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_root / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    failure_indices = [
        i for i, (actual, predicted) in enumerate(zip(targets, predictions, strict=True)) if actual != predicted
    ]
    if max_gallery is not None:
        failure_indices = failure_indices[:max_gallery]
    gallery = []
    gallery_root = output_root / "failure_gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model, target_layer(model, architecture))
    for index in failure_indices:
        record = dataset.records[index]
        stem = f"{index:05d}_{Path(record['image_id']).stem}"
        item = {
            "image_id": record["image_id"],
            "actual": CLASS_NAMES[targets[index]],
            "predicted": CLASS_NAMES[predictions[index]],
            "probabilities": probabilities[index],
            "overlay_status": "error",
        }
        try:
            with Image.open(record["source_path"]) as source:
                original = source.convert("RGB")
                original.save(gallery_root / f"{stem}_original.png")
            tensor, _ = dataset[index]
            input_tensor = tensor[None].to(resolved_device).requires_grad_(True)
            heatmap = cam.generate(input_tensor, predictions[index])[0].detach().cpu().numpy()
            heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").resize(
                original.size, Image.Resampling.BILINEAR
            )
            heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
            color = np.zeros((*heatmap_array.shape, 3), dtype=np.uint8)
            color[..., 0] = np.uint8(255 * heatmap_array)
            color[..., 1] = np.uint8(180 * (1 - np.abs(heatmap_array - 0.5) * 2))
            color[..., 2] = np.uint8(255 * (1 - heatmap_array))
            overlay = Image.blend(original, Image.fromarray(color, mode="RGB"), 0.42)
            overlay_path = gallery_root / f"{stem}_gradcam.png"
            overlay.save(overlay_path)
            item["overlay"] = str(overlay_path)
            item["overlay_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            item["overlay_error"] = str(exc)
        gallery.append(item)
    cam.close()
    (output_root / "failure_cases.json").write_text(json.dumps(gallery, indent=2), encoding="utf-8")
    html = [
        "<html><head><meta charset='utf-8'><title>Failure Case Gallery</title></head><body>",
        f"<h1>{architecture} Failure Case Gallery</h1>",
        f"<p>Failures: {len(gallery)}; overlays: {sum(item['overlay_status'] == 'ok' for item in gallery)}</p>",
        "<table border='1' cellpadding='6'><tr><th>Image</th><th>Actual</th><th>Predicted</th><th>Overlay</th></tr>",
    ]
    for item in gallery:
        overlay = Path(item.get("overlay", "")).name
        html.append(
            f"<tr><td>{item['image_id']}</td><td>{item['actual']}</td><td>{item['predicted']}</td><td>{overlay or item.get('overlay_error', 'missing')}</td></tr>"
        )
    html.extend(["</table></body></html>"])
    (gallery_root / "index.html").write_text("\n".join(html), encoding="utf-8")
    return {
        "architecture": architecture,
        "metrics": metrics,
        "confusion_matrix": confusion.tolist(),
        "failure_count": len(failure_indices),
        "gallery_overlay_count": sum(item["overlay_status"] == "ok" for item in gallery),
        "dataset_version_hash": manifest["dataset_version_hash"],
        "preprocessing_config_hash": manifest["config_hash"],
    }


def promotion_decision(metrics: dict, cam_passed: bool, model_name: str, model_version: str) -> dict:
    reasons = []
    if metrics["macro_f1"] < 0.95:
        reasons.append(f"test macro-F1 {metrics['macro_f1']:.4f} is below 0.95")
    if min(metrics["per_class_f1"]) < 0.90:
        reasons.append(f"minimum per-class F1 {min(metrics['per_class_f1']):.4f} is below 0.90")
    if not cam_passed:
        reasons.append("Grad-CAM sanity check failed")
    return {
        "model_name": model_name,
        "model_version": model_version,
        "decision": "PROMOTED" if not reasons else "HELD_FOR_REVIEW",
        "reason": "passed all evaluation gates" if not reasons else "; ".join(reasons),
        "metrics": metrics,
        "cam_passed": cam_passed,
    }


def gradcam_sanity_check(
    manifest_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    device: str = "cuda",
    controls_per_class: int = 2,
    central_fraction: float = 0.8,
) -> dict:
    """Run a deterministic control-set CAM check using fixed first images per class."""
    resolved_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    model, architecture, _ = load_checkpoint(checkpoint_path, resolved_device)
    dataset = CachedTensorDataset(manifest_path, "testing")
    selected = []
    for class_index in range(len(CLASS_NAMES)):
        selected.extend(
            index for index, record in enumerate(dataset.records) if record["class_name"] == CLASS_NAMES[class_index]
        )
        selected = selected[: controls_per_class * (class_index + 1)]
    cam = GradCAM(model, target_layer(model, architecture))
    control_results = []
    cam_root = output_root / "gradcam_controls"
    cam_root.mkdir(parents=True, exist_ok=True)
    try:
        for index in selected:
            record = dataset.records[index]
            tensor, label = dataset[index]
            input_tensor = tensor[None].to(resolved_device).requires_grad_(True)
            with torch.no_grad():
                predicted = int(model(input_tensor).argmax(1).item())
            heatmap = cam.generate(input_tensor, predicted)[0].detach().cpu().numpy()
            height, width = heatmap.shape
            margin = (1 - central_fraction) / 2
            central = heatmap[
                int(height * margin) : int(height * (1 - margin)), int(width * margin) : int(width * (1 - margin))
            ]
            score = float(central.max() / max(float(heatmap.max()), 1e-8))
            control_results.append(
                {
                    "image_id": record["image_id"],
                    "actual": CLASS_NAMES[label],
                    "predicted": CLASS_NAMES[predicted],
                    "central_activation_score": score,
                    "passed": score >= 0.5,
                }
            )
            plt.imsave(
                cam_root / f"{len(control_results):03d}_{Path(record['image_id']).stem}.png",
                heatmap,
                cmap="magma",
                vmin=0,
                vmax=1,
            )
    finally:
        cam.close()
    result = {
        "architecture": architecture,
        "controls": control_results,
        "threshold": 0.5,
        "passed": all(item["passed"] for item in control_results),
    }
    (output_root / "gradcam_sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
