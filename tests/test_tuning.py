from pathlib import Path

import optuna

from app.tuning.studies import create_study, ensure_storage_parent, suggest_parameters


def test_search_space_constrains_fine_tune_learning_rate() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3), direction="maximize")
    trial = study.ask()
    parameters = suggest_parameters(trial)

    assert 1e-5 <= parameters["head_learning_rate"] <= 1e-3
    assert 1e-6 <= parameters["fine_tune_learning_rate"] <= 1e-4
    assert parameters["fine_tune_learning_rate"] <= parameters["head_learning_rate"]
    assert parameters["batch_size"] in (16, 32, 64)
    assert parameters["unfreeze_depth"] in (1, 2, 3)


def test_architecture_study_is_resumable_and_separate(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    ensure_storage_parent(storage)
    first = create_study("resnet50", storage)
    second = create_study("resnet50", storage)
    other = create_study("convnext_tiny", storage)

    assert first.study_name == second.study_name
    assert first.study_name != other.study_name
    assert second.study_name == "brainseg-resnet50"
