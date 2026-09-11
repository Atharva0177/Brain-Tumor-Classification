from pathlib import Path
from unittest.mock import Mock, patch

from app.tuning.studies import promote_checkpoint_to_staging


def test_checkpoint_promotion_requires_registry_and_preserves_tags(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trial_0.pt"
    summary = {
        "architecture": "resnet50",
        "study_name": "brainseg-resnet50",
        "best_value": 0.75,
        "best_params": {},
        "dataset_version_hash": "dataset",
        "preprocessing_config_hash": "preprocess",
    }
    checkpoint.write_bytes(b"placeholder")
    run = _NullContext()
    run.info = Mock(run_id="run-1")
    with (
        patch("app.tuning.studies.torch.load", return_value={"architecture": "resnet50", "model": {}, "params": {}}),
        patch("app.tuning.studies.build_model") as build_model,
        patch("app.tuning.studies.mlflow.start_run", return_value=run),
        patch("app.tuning.studies.mlflow.log_params"),
        patch("app.tuning.studies.mlflow.log_metric"),
        patch("app.tuning.studies.mlflow.log_artifact"),
        patch("app.tuning.studies.mlflow.pytorch", Mock()),
        patch("app.tuning.studies.mlflow.register_model", return_value=Mock(version="1")),
        patch("app.tuning.studies.mlflow.tracking.MlflowClient") as client_factory,
    ):
        model = build_model.return_value
        model.load_state_dict = Mock()
        client_factory.return_value = Mock()
        result = promote_checkpoint_to_staging(checkpoint, "brainseg-resnet50", summary, "http://mlflow")

    assert result["stage"] == "Staging"
    assert result["tags"]["dataset_version_hash"] == "dataset"
    client_factory.return_value.transition_model_version_stage.assert_called_once()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
