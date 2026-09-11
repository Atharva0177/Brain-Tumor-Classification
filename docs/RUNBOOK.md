# BrainSeg Operations Runbook

## Start the stack

```powershell
conda activate brainseg
docker compose up -d postgres redis mlflow api celery-worker
```

Services:

| Service | URL |
|---|---|
| API | `http://localhost:59000` |
| API docs | `http://localhost:59000/docs` |
| MLflow | `http://localhost:55000` |

Check status:

```powershell
docker compose ps
Invoke-WebRequest -UseBasicParsing http://localhost:59000/health
```

## Production model

The API loads `brainseg-convnext-base` from MLflow `Production`. Confirm lineage:

```powershell
curl.exe http://localhost:59000/model-info
```

Never select a checkpoint manually for serving. Promote through the evaluation gate and registry.

## Inference

```powershell
curl.exe -X POST `
  -F "file=@C:\path\to\scan.jpg" `
  http://localhost:59000/predict
```

The response includes probabilities, predicted class, model version, lineage hashes, and a base64 PNG Grad-CAM overlay.

## Shutdown

```powershell
```

Do not use `docker compose down -v` unless deleting local database, registry, Redis, and MLflow data is intentional.

## Troubleshooting

- API `503`: inspect `docker compose logs api`; confirm MLflow Production model and shared `mlflow-data` volume.
- Port conflict: check Windows excluded ranges with `netsh interface ipv4 show excludedportrange protocol=tcp`; override `API_PORT` or other host ports in `.env`.
- MLflow invalid Host header: rebuild MLflow after changing `infra/mlflow/entrypoint.py` allowed hosts.
- Tests: use `python -m pytest`, not a stale global `pytest.exe` launcher.
