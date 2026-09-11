from unittest.mock import patch

from app.datasets.acquisition import DatasetAcquisitionError
from worker.tasks import PipelineTask, acquire_dataset_task


def test_download_task_retries_only_dataset_acquisition_errors() -> None:
    assert DatasetAcquisitionError in acquire_dataset_task.autoretry_for
    assert acquire_dataset_task.max_retries == 3
    assert acquire_dataset_task.retry_backoff is True
    assert ValueError not in acquire_dataset_task.autoretry_for


def test_download_task_preserves_failure_for_non_retryable_errors() -> None:
    with patch("worker.tasks.acquire_dataset", side_effect=ValueError("database failure")):
        result = acquire_dataset_task.apply()

    assert result.failed()
    assert isinstance(result.result, ValueError)
    assert "database failure" in str(result.result)


def test_exhausted_task_marks_pipeline_failed(monkeypatch) -> None:
    class FakeSession:
        def __init__(self):
            self.run = type("Run", (), {"status": "RUNNING", "error_detail": None})()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, model, run_id):
            return self.run if run_id == "run-1" else None

    session = FakeSession()
    monkeypatch.setattr("worker.tasks.SessionLocal.begin", lambda: session)

    PipelineTask().on_failure(RuntimeError("download failed"), "task-1", (), {"pipeline_run_id": "run-1"}, None)

    assert session.run.status == "FAILED"
    assert "download failed" in session.run.error_detail
