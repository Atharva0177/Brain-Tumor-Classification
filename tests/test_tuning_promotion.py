from unittest.mock import Mock, patch

import optuna

from app.tuning.studies import promote_best_trial_to_staging


def test_promotion_requires_completed_trial() -> None:
    study = optuna.create_study(direction="maximize")
    summary = {"architecture": "resnet50", "dataset_version_hash": "data", "preprocessing_config_hash": "config"}

    try:
        promote_best_trial_to_staging(study, "brainseg-resnet50", summary)
    except ValueError as exc:
        assert "completed best trial" in str(exc)
    else:
        raise AssertionError("promotion unexpectedly succeeded")


def test_promotion_registers_model_and_sets_lineage_tags() -> None:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr("mlflow_run_id", "run-1")
    study.tell(trial, 0.8)
    summary = {"architecture": "resnet50", "dataset_version_hash": "data", "preprocessing_config_hash": "config"}
    version = Mock(version="3")
    client = Mock()
    with (
        patch("app.tuning.studies.mlflow.register_model", return_value=version),
        patch("app.tuning.studies.mlflow.tracking.MlflowClient", return_value=client),
    ):
        result = promote_best_trial_to_staging(study, "brainseg-resnet50", summary)

    assert result["model_version"] == "3"
    client.transition_model_version_stage.assert_called_once()
    assert client.set_model_version_tag.call_count == 3
