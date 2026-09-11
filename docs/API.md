# API Contract

Base URL: `http://localhost:59000`

## `GET /health/live`

Returns process liveness without dependency checks.

```json
{"status":"ok"}
```

## `GET /health`

Returns HTTP `200` only when PostgreSQL, Redis, MLflow, and a Production model are ready. Otherwise returns HTTP `503`.

## `GET /model-info`

Returns the active registry model, architecture, version, training run, dataset/preprocessing hashes, and evaluation metrics.

## `POST /predict`

Accepts multipart field `file` containing an image. Returns:

```json
{
  "predicted_class": "glioma",
  "confidence": 0.91,
  "probabilities": {"glioma": 0.91, "meningioma": 0.03, "notumor": 0.02, "pituitary": 0.04},
  "gradcam_overlay_base64": "...",
  "model": {
    "model_name": "brainseg-convnext-base",
    "model_version": "1",
    "architecture": "convnext_base",
    "training_run_id": "...",
    "dataset_version_hash": "...",
    "preprocessing_config_hash": "...",
    "evaluation_metrics": {}
  }
}
```

Invalid non-image uploads return `415`; malformed images return `400`; no loaded Production model returns `503`.
