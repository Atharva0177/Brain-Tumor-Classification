"""Create Phase 0 pipeline lineage tables.

Revision ID: 0001_phase0_lineage
Revises:
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_phase0_lineage"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "brainseg"


def upgrade() -> None:
    op.execute(sa.text("CREATE SCHEMA IF NOT EXISTS brainseg"))
    op.execute(sa.text("CREATE SCHEMA IF NOT EXISTS optuna"))

    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error_detail", sa.Text()),
        schema=SCHEMA,
    )
    op.create_table(
        "pipeline_stages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("pipeline_run_id", sa.String(36), sa.ForeignKey(f"{SCHEMA}.pipeline_runs.id"), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("input_ids", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("config_hash", sa.String(64)),
        sa.Column("artifact_uri", sa.Text()),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("error_detail", sa.Text()),
        schema=SCHEMA,
    )
    op.create_index("ix_pipeline_stages_run", "pipeline_stages", ["pipeline_run_id"], schema=SCHEMA)
    op.create_table(
        "dataset_versions",
        sa.Column("version_hash", sa.String(64), primary_key=True),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("archive_uri", sa.Text()),
        sa.Column("manifest_uri", sa.Text()),
        sa.Column("file_count", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema=SCHEMA,
    )
    op.create_table(
        "verification_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("dataset_version_hash", sa.String(64), sa.ForeignKey(f"{SCHEMA}.dataset_versions.version_hash"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("report_uri", sa.Text()),
        sa.Column("report_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema=SCHEMA,
    )
    op.create_table(
        "preprocessing_configs",
        sa.Column("config_hash", sa.String(64), primary_key=True),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema=SCHEMA,
    )
    op.create_table(
        "split_assignments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("dataset_version_hash", sa.String(64), sa.ForeignKey(f"{SCHEMA}.dataset_versions.version_hash"), nullable=False),
        sa.Column("config_hash", sa.String(64), sa.ForeignKey(f"{SCHEMA}.preprocessing_configs.config_hash"), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("assignment_uri", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema=SCHEMA,
    )
    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("pipeline_run_id", sa.String(36), sa.ForeignKey(f"{SCHEMA}.pipeline_runs.id"), nullable=False),
        sa.Column("dataset_version_hash", sa.String(64), sa.ForeignKey(f"{SCHEMA}.dataset_versions.version_hash"), nullable=False),
        sa.Column("config_hash", sa.String(64), sa.ForeignKey(f"{SCHEMA}.preprocessing_configs.config_hash"), nullable=False),
        sa.Column("architecture", sa.String(64), nullable=False),
        sa.Column("mlflow_run_id", sa.String(128)),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema=SCHEMA,
    )
    op.create_table(
        "promotion_decisions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("training_run_id", sa.String(36), sa.ForeignKey(f"{SCHEMA}.training_runs.id"), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema=SCHEMA,
    )
    op.create_table(
        "prediction_lineage",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("training_run_id", sa.String(36), sa.ForeignKey(f"{SCHEMA}.training_runs.id")),
        sa.Column("dataset_version_hash", sa.String(64), sa.ForeignKey(f"{SCHEMA}.dataset_versions.version_hash")),
        sa.Column("config_hash", sa.String(64), sa.ForeignKey(f"{SCHEMA}.preprocessing_configs.config_hash")),
        sa.Column("input_hash", sa.String(64)),
        sa.Column("predicted_class", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        schema=SCHEMA,
    )


def downgrade() -> None:
    for table in [
        "prediction_lineage",
        "promotion_decisions",
        "training_runs",
        "split_assignments",
        "preprocessing_configs",
        "verification_reports",
        "dataset_versions",
        "pipeline_stages",
        "pipeline_runs",
    ]:
        op.drop_table(table, schema=SCHEMA)
