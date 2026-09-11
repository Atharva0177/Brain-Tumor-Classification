from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.training.metrics import classification_metrics
from app.training.models import (
    ModelSpec,
    build_model,
    freeze_for_head_warmup,
    trainable_parameter_count,
    unfreeze_last_stages,
)
from app.training.trainer import TrainingConfig, train_model


def test_supported_models_return_four_class_logits_without_downloads() -> None:
    for architecture in ("convnext_tiny", "convnext_base", "resnet50", "efficientnet_b0"):
        model = build_model(ModelSpec(architecture, pretrained=False))
        logits = model(torch.randn(2, 3, 224, 224))
        assert logits.shape == (2, 4)


def test_freeze_and_unfreeze_plans_are_inspectable() -> None:
    model = build_model(ModelSpec("resnet50", pretrained=False))
    freeze_for_head_warmup(model, "resnet50")
    head_count = trainable_parameter_count(model)
    unfreeze_last_stages(model, "resnet50", depth=2)

    assert head_count > 0
    assert trainable_parameter_count(model) > head_count


def test_metrics_include_macro_and_per_class_scores() -> None:
    metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["per_class_f1"]) == 4


def test_cpu_trainer_writes_best_checkpoint_and_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.training.trainer.mlflow.start_run", lambda **kwargs: _NullContext())
    monkeypatch.setattr("app.training.trainer.mlflow.log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.training.trainer.mlflow.log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.training.trainer.mlflow.log_artifact", lambda *args, **kwargs: None)
    inputs = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
    model = build_model(ModelSpec("resnet50", pretrained=False))
    config = TrainingConfig(
        "resnet50",
        "dataset",
        "preprocess",
        "split",
        head_warmup_epochs=1,
        fine_tune_epochs=1,
        amp=False,
        device="cpu",
        patience=3,
    )

    summary = train_model(model, loader, loader, config, tmp_path)

    assert summary["best_epoch"] >= 0
    assert (tmp_path / "resnet50_best.pt").is_file()
    assert (tmp_path / "resnet50_training_summary.json").is_file()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
