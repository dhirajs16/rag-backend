"""create interview bookings table

Revision ID: b4172e18eac1
Revises: 6d7388fab1bb
Create Date: 2026-11-14 13:05:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4172e18eac1"
down_revision: Union[str, Sequence[str], None] = "6d7388fab1bb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create durable storage for confirmed interview bookings."""
    op.create_table(
        "interview_bookings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("scheduled_date", sa.Date(), nullable=False),
        sa.Column("scheduled_time", sa.Time(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scheduled_date",
            "scheduled_time",
            name="uq_interview_booking_slot",
        ),
    )
    op.create_index(
        op.f("ix_interview_bookings_conversation_id"),
        "interview_bookings",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_interview_bookings_email"),
        "interview_bookings",
        ["email"],
        unique=False,
    )


def downgrade() -> None:
    """Remove interview booking storage."""
    op.drop_index(
        op.f("ix_interview_bookings_email"),
        table_name="interview_bookings",
    )
    op.drop_index(
        op.f("ix_interview_bookings_conversation_id"),
        table_name="interview_bookings",
    )
    op.drop_table("interview_bookings")
