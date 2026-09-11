from pathlib import Path
from uuid import uuid4

from celery import Task

from app.datasets.acquisition import DatasetAcquisitionError, acquire_dataset
from app.datasets.verification import verify_dataset, write_verification_artifacts
from app.db.models import DatasetVersion, PipelineRun, VerificationReport
from app.db.session import SessionLocal
from app.observability import get_logger, task_context
from app.tuning.studies import run_study
from worker.celery_app import celery_app

logger = get_logger("tasks")


class PipelineTask(Task):
    """Log terminal failures and propagate them to an existing pipeline run."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        pipeline_run_id = kwargs.get("pipeline_run_id") if kwargs else None
        task_name = getattr(self, "name", None) or "UNKNOWN"
        context = task_context(task_id, task_name.rsplit(".", 1)[-1].upper())
        logger.error("task_exhausted_failure error_type=%s", type(exc).__name__, extra=context)
        if pipeline_run_id:
            try:
                with SessionLocal.begin() as session:
                    run = session.get(PipelineRun, pipeline_run_id)
                    if run:
                        run.status = "FAILED"
                        run.error_detail = f"{type(exc).__name__}: {exc}"
            except Exception:  # noqa: BLE001
                logger.exception("pipeline_failure_persistence_failed", extra=context)
        return super().on_failure(exc, task_id, args, kwargs, einfo)


@celery_app.task(name="brainseg.smoke")
def smoke_task() -> dict[str, str]:
    """Verify that a worker can accept and complete a task."""
    logger.info("task_succeeded", extra=task_context(None, "SMOKE"))
    return {"status": "ok", "task": "smoke"}


@celery_app.task(
    bind=True,
    base=PipelineTask,
    name="brainseg.acquire_dataset",
    autoretry_for=(DatasetAcquisitionError,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=False,
    max_retries=3,
)
def acquire_dataset_task(
    self,
    dataset: str | None = None,
    archive_path: str | None = None,
    pipeline_run_id: str | None = None,
) -> dict:
    context = task_context(self.request.id, "DOWNLOAD")
    logger.info("task_started", extra=context)
    try:
        result = acquire_dataset(dataset=dataset, archive_path=Path(archive_path) if archive_path else None)
        with SessionLocal.begin() as session:
            session.merge(
                DatasetVersion(
                    version_hash=result["dataset_version_hash"],
                    source=dataset or "kaggle",
                    status="COMPLETE",
                    archive_uri=result.get("archive"),
                    manifest_uri=result.get("manifest_path"),
                    file_count=len(result["manifest"]),
                )
            )
        logger.info("task_succeeded status=%s", result["status"], extra=context)
        return result
    except DatasetAcquisitionError:
        logger.warning("task_retryable_failure retries=%s", self.request.retries, extra=context)
        raise
    except Exception:
        logger.exception("task_failed", extra=context)
        raise


@celery_app.task(bind=True, name="brainseg.verify_dataset")
def verify_dataset_task(self, dataset_root: str, dataset_version_hash: str | None = None) -> dict:
    context = task_context(self.request.id, "VERIFY")
    logger.info("task_started", extra=context)
    try:
        report = verify_dataset(Path(dataset_root), dataset_version_hash=dataset_version_hash)
        destination = Path("artifacts") / "verification" / (dataset_version_hash or uuid4().hex)
        json_path, markdown_path = write_verification_artifacts(report, destination)
        if dataset_version_hash:
            with SessionLocal.begin() as session:
                session.add(
                    VerificationReport(
                        id=str(uuid4()),
                        dataset_version_hash=dataset_version_hash,
                        status=report["status"],
                        report_uri=str(json_path),
                        report_json=report,
                    )
                )
        report["json_artifact"] = str(json_path)
        report["markdown_artifact"] = str(markdown_path)
        logger.info("task_succeeded status=%s", report["status"], extra=context)
        return report
    except Exception:
        logger.exception("task_failed", extra=context)
        raise


@celery_app.task(bind=True, name="brainseg.tune_model")
def tune_model_task(
    self, architecture: str, manifest_path: str, n_trials: int = 40, storage: str | None = None
) -> dict:
    context = task_context(self.request.id, "TUNE")
    logger.info("task_started architecture=%s trials=%s", architecture, n_trials, extra=context)
    try:
        result = run_study(
            architecture,
            Path(manifest_path),
            Path("artifacts") / "tuning" / architecture,
            n_trials=n_trials,
            storage=storage,
        )
        logger.info("task_succeeded best_value=%s", result.get("best_value"), extra=context)
        return result
    except Exception:
        logger.exception("task_failed", extra=context)
        raise
