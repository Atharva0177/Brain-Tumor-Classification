from pathlib import Path

from app.evaluation.gate import target_layer_path, validate_gradcam_target, write_explainability_review_metadata
from app.training.models import ModelSpec, build_model


def test_gradcam_targets_validate_for_all_architectures() -> None:
    for architecture in ("convnext_tiny", "convnext_base", "resnet50", "efficientnet_b0"):
        model = build_model(ModelSpec(architecture, pretrained=False))
        result = validate_gradcam_target(model, architecture, (1, 3, 64, 64))
        assert result["target_layer"] == target_layer_path(architecture)
        assert len(result["activation_shape"]) == 4


def test_explainability_metadata_is_immutable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"model")
    result = {"passed": True, "controls": [], "threshold": 0.5}
    path = write_explainability_review_metadata(
        tmp_path / "review", "model", "1", "convnext_base", checkpoint, "dataset", "preprocess", result
    )
    assert path.is_file()
    assert (
        write_explainability_review_metadata(
            tmp_path / "review", "model", "1", "convnext_base", checkpoint, "dataset", "preprocess", result
        )
        == path
    )
    checkpoint.write_bytes(b"changed")
    try:
        write_explainability_review_metadata(
            tmp_path / "review", "model", "1", "convnext_base", checkpoint, "dataset", "preprocess", result
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("metadata was mutable")
