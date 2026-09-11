"""Add dataset verification constraints.

Revision ID: 0002_dataset_verification
Revises: 0001_phase0_lineage
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002_dataset_verification"
down_revision: Union[str, Sequence[str], None] = "0001_phase0_lineage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_dataset_versions_status",
        "dataset_versions",
        "status IN ('COMPLETE', 'FAILED', 'DOWNLOADING')",
        schema="brainseg",
    )
    op.create_check_constraint(
        "ck_verification_reports_status",
        "verification_reports",
        "status IN ('COMPLETE', 'VERIFICATION_FAILED')",
        schema="brainseg",
    )


def downgrade() -> None:
    op.drop_constraint("ck_verification_reports_status", "verification_reports", schema="brainseg", type_="check")
    op.drop_constraint("ck_dataset_versions_status", "dataset_versions", schema="brainseg", type_="check")
