from worker.tasks import smoke_task


def test_smoke_task_returns_success_payload() -> None:
    assert smoke_task.run() == {"status": "ok", "task": "smoke"}
