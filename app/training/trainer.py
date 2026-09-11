from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.training.metrics import classification_metrics
from app.training.models import freeze_for_head_warmup, trainable_parameter_count, unfreeze_last_stages


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str
    dataset_version_hash: str
    preprocessing_config_hash: str
    split_id: str
    head_warmup_epochs: int = 1
    fine_tune_epochs: int = 1
    head_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    unfreeze_depth: int = 2
    patience: int = 3
    amp: bool = True
    device: str = "cuda"


def _run_epoch(model, loader, loss_function, optimizer, scaler, device, training: bool) -> tuple[float, dict]:
    model.train(training)
    total_loss = 0.0
    predictions, targets = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(inputs)
            loss = loss_function(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach()) * inputs.size(0)
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    metrics = classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics["loss"], metrics


def _save_checkpoint(path: Path, model, optimizer, scaler, config: TrainingConfig, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def train_model(
    model, train_loader: DataLoader, validation_loader: DataLoader, config: TrainingConfig, checkpoint_root: Path
) -> dict:
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    loss_function = nn.CrossEntropyLoss()
    history = []
    best_metric = float("-inf")
    best_epoch = -1
    stale_epochs = 0
    started = time.perf_counter()
    with mlflow.start_run(run_name=f"baseline-{config.architecture}"):
        mlflow.log_params(
            {**asdict(config), "resolved_device": str(device), "trainable_parameters": trainable_parameter_count(model)}
        )
        for phase, epochs, learning_rate in (
            ("head_warmup", config.head_warmup_epochs, config.head_learning_rate),
            ("fine_tune", config.fine_tune_epochs, config.fine_tune_learning_rate),
        ):
            if phase == "head_warmup":
                freeze_for_head_warmup(model, config.architecture)
            else:
                unfreeze_last_stages(model, config.architecture, config.unfreeze_depth)
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=learning_rate,
                weight_decay=config.weight_decay,
            )
            for _phase_epoch in range(epochs):
                epoch = len(history)
                train_loss, train_metrics = _run_epoch(
                    model, train_loader, loss_function, optimizer, scaler, device, True
                )
                validation_loss, validation_metrics = _run_epoch(
                    model, validation_loader, loss_function, optimizer, scaler, device, False
                )
                record = {
                    "epoch": epoch,
                    "phase": phase,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                }
                history.append(record)
                mlflow.log_metrics(
                    {key: value for key, value in record.items() if isinstance(value, (int, float))}, step=epoch
                )
                if validation_metrics["macro_f1"] > best_metric:
                    best_metric = validation_metrics["macro_f1"]
                    best_epoch = epoch
                    stale_epochs = 0
                    _save_checkpoint(
                        checkpoint_root / f"{config.architecture}_best.pt",
                        model,
                        optimizer,
                        scaler,
                        config,
                        epoch,
                        record,
                    )
                else:
                    stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
        duration = time.perf_counter() - started
        summary = {
            "architecture": config.architecture,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_metric,
            "duration_seconds": duration,
            "history": history,
        }
        summary_path = checkpoint_root / f"{config.architecture}_training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))
    return summary
