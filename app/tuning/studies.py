from __future__ import annotations

import json
from pathlib import Path

import mlflow
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.config import settings
from app.training.data import CachedTensorDataset
from app.training.models import (
    ModelSpec,
    build_model,
    freeze_for_head_warmup,
    unfreeze_last_stages,
)
from app.training.trainer import _run_epoch


def optuna_storage_url() -> str:
    """Use the dedicated Postgres schema when available, with SQLite for local smoke runs."""
    return (
        f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password.get_secret_value()}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        "?options=-csearch_path%3Doptuna"
    )


def ensure_storage_parent(storage: str) -> None:
    if storage.startswith("sqlite:///"):
        database_path = Path(storage.removeprefix("sqlite:///"))
        database_path.parent.mkdir(parents=True, exist_ok=True)


def suggest_parameters(trial: optuna.Trial) -> dict:
    head_lr = trial.suggest_float("head_learning_rate", 1e-5, 1e-3, log=True)
    fine_lr = trial.suggest_float("fine_tune_learning_rate", 1e-6, min(1e-4, head_lr), log=True)
    return {
        "head_learning_rate": head_lr,
        "fine_tune_learning_rate": fine_lr,
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "unfreeze_depth": trial.suggest_categorical("unfreeze_depth", [1, 2, 3]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "head_warmup_epochs": trial.suggest_int("head_warmup_epochs", 2, 5),
        "fine_tune_epochs": trial.suggest_int("fine_tune_epochs", 5, 15),
        "augmentation_strength": trial.suggest_categorical("augmentation_strength", ["light", "medium", "heavy"]),
    }


def _trial_objective(
    trial: optuna.Trial,
    architecture: str,
    manifest_path: Path,
    checkpoint_root: Path,
    training_context: dict,
    device_name: str,
    pretrained: bool,
    trial_run_id: str,
) -> float:
    parameters = suggest_parameters(trial)
    mlflow.set_tag("optuna_trial_number", str(trial.number))
    trial.set_user_attr("mlflow_run_id", trial_run_id)
    mlflow.log_params({**parameters, **training_context, "architecture": architecture, "device": device_name})
    train_dataset = CachedTensorDataset(manifest_path, "train")
    validation_dataset = CachedTensorDataset(manifest_path, "validation")
    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_dataset, batch_size=parameters["batch_size"], shuffle=False, num_workers=0
    )
    model = build_model(ModelSpec(architecture, pretrained=pretrained))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_function = nn.CrossEntropyLoss()
    best = float("-inf")
    epoch_number = 0
    try:
        phases = (
            ("head_warmup", parameters["head_warmup_epochs"], parameters["head_learning_rate"]),
            ("fine_tune", parameters["fine_tune_epochs"], parameters["fine_tune_learning_rate"]),
        )
        for phase, epochs, learning_rate in phases:
            if phase == "head_warmup":
                freeze_for_head_warmup(model, architecture)
            else:
                unfreeze_last_stages(model, architecture, parameters["unfreeze_depth"])
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=learning_rate,
                weight_decay=parameters["weight_decay"],
            )
            for _ in range(epochs):
                _, train_metrics = _run_epoch(model, train_loader, loss_function, optimizer, scaler, device, True)
                _, validation_metrics = _run_epoch(
                    model, validation_loader, loss_function, optimizer, scaler, device, False
                )
                score = float(validation_metrics["macro_f1"])
                trial.report(score, step=epoch_number)
                mlflow.log_metrics(
                    {"train_macro_f1": train_metrics["macro_f1"], "validation_macro_f1": score}, step=epoch_number
                )
                best = max(best, score)
                epoch_number += 1
                if trial.should_prune():
                    mlflow.set_tag("trial_status", "pruned")
                    raise optuna.TrialPruned()
        checkpoint = checkpoint_root / f"trial_{trial.number}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "architecture": architecture, "params": parameters}, checkpoint)
        mlflow.log_artifact(str(checkpoint))
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.set_tag("trial_status", "complete")
        return best
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mlflow.set_tag("trial_status", "oom")
        raise RuntimeError(f"CUDA OOM: {exc}") from exc


def create_study(architecture: str, storage: str, load_if_exists: bool = True) -> optuna.Study:
    return optuna.create_study(
        study_name=f"brainseg-{architecture}",
        storage=storage,
        load_if_exists=load_if_exists,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )


def run_study(
    architecture: str,
    manifest_path: Path,
    checkpoint_root: Path,
    n_trials: int = 40,
    storage: str | None = None,
    device: str = "cuda",
    pretrained: bool = True,
    tracking_uri: str | None = None,
) -> dict:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    storage = storage or optuna_storage_url()
    ensure_storage_parent(storage)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context = {
        "dataset_version_hash": manifest["dataset_version_hash"],
        "preprocessing_config_hash": manifest["config_hash"],
        "split_id": manifest["config_hash"],
        "study_budget_estimate": 40,
        "requested_trial_budget": n_trials,
    }
    study = create_study(architecture, storage)
    parent_name = f"sweep-{architecture}"
    with mlflow.start_run(run_name=parent_name):
        mlflow.log_params(
            {
                **context,
                "architecture": architecture,
                "study_name": study.study_name,
                "storage": storage,
                "device": device,
            }
        )

        def objective(trial: optuna.Trial) -> float:
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                return _trial_objective(
                    trial,
                    architecture,
                    manifest_path,
                    checkpoint_root,
                    context,
                    device,
                    pretrained,
                    mlflow.active_run().info.run_id,
                )

        study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
        summary = {
            "architecture": architecture,
            "study_name": study.study_name,
            "study_id": study.study_name,
            "requested_trials": n_trials,
            "completed_trials": len(study.trials),
            "best_trial": study.best_trial.number if study.best_trial else None,
            "best_value": study.best_value if study.best_trial else None,
            "best_params": study.best_params if study.best_trial else {},
            "trial_states": {
                state.name: sum(t.state == state for t in study.trials) for state in optuna.trial.TrialState
            },
            "dataset_version_hash": manifest["dataset_version_hash"],
            "preprocessing_config_hash": manifest["config_hash"],
        }
        summary_path = checkpoint_root / f"{architecture}_study_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))
    return summary


def promote_best_trial_to_staging(study: optuna.Study, model_name: str, summary: dict) -> dict:
    """Register the best completed trial's MLflow model and move it to Staging."""
    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise ValueError("A completed best trial is required for Staging promotion") from exc
    if best_trial is None or best_trial.state != optuna.trial.TrialState.COMPLETE:
        raise ValueError("A completed best trial is required for Staging promotion")
    run_id = best_trial.user_attrs.get("mlflow_run_id")
    if not run_id:
        raise ValueError("Best trial does not have an MLflow run ID")
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        model_name, model_version.version, stage="Staging", archive_existing_versions=False
    )
    result = {
        "model_name": model_name,
        "model_version": model_version.version,
        "stage": "Staging",
        "training_run_id": run_id,
        "study_name": study.study_name,
        "trial_number": best_trial.number,
        "dataset_version_hash": summary["dataset_version_hash"],
        "preprocessing_config_hash": summary["preprocessing_config_hash"],
    }
    client.set_model_version_tag(model_name, model_version.version, "architecture", summary["architecture"])
    client.set_model_version_tag(
        model_name, model_version.version, "dataset_version_hash", summary["dataset_version_hash"]
    )
    client.set_model_version_tag(
        model_name, model_version.version, "preprocessing_config_hash", summary["preprocessing_config_hash"]
    )
    return result


def promote_checkpoint_to_staging(
    checkpoint_path: Path,
    model_name: str,
    summary: dict,
    tracking_uri: str,
) -> dict:
    """Publish an existing completed trial checkpoint to a live MLflow registry."""
    mlflow.set_tracking_uri(tracking_uri)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = build_model(ModelSpec(architecture, pretrained=False))
    model.load_state_dict(checkpoint["model"])
    parameters = checkpoint.get("params", summary.get("best_params", {}))
    trial_number = checkpoint_path.stem.removeprefix("trial_")
    with mlflow.start_run(run_name=f"registry-promotion-{architecture}") as run:
        mlflow.log_params(
            {
                **parameters,
                "architecture": architecture,
                "study_name": summary["study_name"],
                "optuna_trial_number": trial_number,
                "dataset_version_hash": summary["dataset_version_hash"],
                "preprocessing_config_hash": summary["preprocessing_config_hash"],
                "split_id": summary["preprocessing_config_hash"],
            }
        )
        mlflow.log_metric("validation_macro_f1", float(summary["best_value"]))
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoint")
        mlflow.pytorch.log_model(model, artifact_path="model")
        run_id = run.info.run_id
    model_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        model_name, model_version.version, stage="Staging", archive_existing_versions=False
    )
    tags = {
        "architecture": architecture,
        "study_name": summary["study_name"],
        "optuna_trial_number": trial_number,
        "dataset_version_hash": summary["dataset_version_hash"],
        "preprocessing_config_hash": summary["preprocessing_config_hash"],
        "validation_macro_f1": str(summary["best_value"]),
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, model_version.version, key, value)
    return {
        "model_name": model_name,
        "model_version": model_version.version,
        "stage": "Staging",
        "training_run_id": run_id,
        "tags": tags,
    }
