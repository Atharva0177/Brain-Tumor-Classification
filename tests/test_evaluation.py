from app.evaluation.gate import promotion_decision


def test_promotion_gate_holds_model_below_thresholds() -> None:
    decision = promotion_decision({"macro_f1": 0.8, "per_class_f1": [0.95, 0.95, 0.95, 0.95]}, True, "model", "1")
    assert decision["decision"] == "HELD_FOR_REVIEW"
    assert "macro-F1" in decision["reason"]


def test_promotion_gate_requires_gradcam() -> None:
    decision = promotion_decision({"macro_f1": 0.96, "per_class_f1": [0.91, 0.92, 0.95, 0.94]}, False, "model", "1")
    assert decision["decision"] == "HELD_FOR_REVIEW"
    assert "Grad-CAM" in decision["reason"]


def test_gallery_metadata_requires_overlay_status() -> None:
    item = {"image_id": "Testing/glioma/a.jpg", "actual": "glioma", "predicted": "meningioma", "overlay_status": "ok"}
    assert item["overlay_status"] in {"ok", "error"}
