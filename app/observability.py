import logging


def get_logger(name: str) -> logging.Logger:
    """Return the project logger; callers attach only non-secret identifiers."""
    return logging.getLogger(f"brainseg.{name}")


def task_context(task_id: str | None, stage: str) -> dict[str, str]:
    return {"task_id": task_id or "unknown", "stage": stage}
