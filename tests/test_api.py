from fastapi.testclient import TestClient

from app import main


def test_liveness_does_not_require_dependencies() -> None:
    client = TestClient(main.app)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_service_unavailable_when_dependency_is_down(monkeypatch) -> None:
    monkeypatch.setattr(
        main, "dependency_health", lambda: {"status": "unavailable", "checks": {"redis": {"status": "unavailable"}}}
    )
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_model_info_returns_loaded_lineage(monkeypatch) -> None:
    monkeypatch.setattr(
        main.model_service,
        "info",
        lambda: {"model_name": "brainseg-convnext-base", "model_version": "1", "architecture": "convnext_base"},
    )
    client = TestClient(main.app)

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_version"] == "1"


def test_predict_rejects_non_image_upload(monkeypatch) -> None:
    monkeypatch.setattr(main.model_service, "status", lambda: {"loaded": True})
    client = TestClient(main.app)

    response = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})

    assert response.status_code == 415
