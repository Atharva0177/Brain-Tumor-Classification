# Architecture

```text
Kaggle / ZIP
    -> data/raw/<dataset_hash>
    -> verification report + Postgres lineage
    -> data/processed/<config_hash>
    -> train/validation split + untouched Testing
    -> PyTorch baselines / Optuna studies
    -> MLflow registry Staging
    -> held-out evaluation + Grad-CAM gate
    -> MLflow Production
    -> FastAPI loader and atomic polling reload
    -> /predict API
```

Runtime services are isolated under the `btc-classification` Compose project. Host ports are intentionally separate from the existing `brainseg` deployment.

## Current Production Lineage

- Model: `brainseg-convnext-base` version `1`
- Architecture: ConvNeXt-Base
- Dataset: `974e32f0fa628a307ba1bc2ec7f48d137b8190da0582678d7c5c75c8120bf383`
- Preprocessing: `7d91c97fccc3d9b58cbb353d373584651e6a2f21c5d67262f0064c78316611d8`
- Evaluation macro-F1: `0.963874`
- Minimum per-class F1: `0.928759`
