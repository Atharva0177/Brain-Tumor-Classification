from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = {"schema": "brainseg"}

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    error_detail: Mapped[str | None] = mapped_column(Text())


class PipelineStage(Base):
    __tablename__ = "pipeline_stages"
    __table_args__ = {"schema": "brainseg"}

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    pipeline_run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_ids: Mapped[dict] = mapped_column(JSON, default=dict)
    config_hash: Mapped[str | None] = mapped_column(String(64))
    artifact_uri: Mapped[str | None] = mapped_column(Text())
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_detail: Mapped[str | None] = mapped_column(Text())


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = {"schema": "brainseg"}

    version_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    archive_uri: Mapped[str | None] = mapped_column(Text())
    manifest_uri: Mapped[str | None] = mapped_column(Text())
    file_count: Mapped[int | None] = mapped_column(Integer())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class VerificationReport(Base):
    __tablename__ = "verification_reports"
    __table_args__ = {"schema": "brainseg"}

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_version_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    report_uri: Mapped[str | None] = mapped_column(Text())
    report_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
