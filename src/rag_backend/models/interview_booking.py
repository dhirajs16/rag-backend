"""Interview booking persistence model."""

import uuid
from datetime import date, datetime, time

from sqlalchemy import Date, DateTime, String, Time, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from rag_backend.db.base import Base


class InterviewBooking(Base):
    """A confirmed interview time captured through the chat API."""

    __tablename__ = "interview_bookings"
    __table_args__ = (
        UniqueConstraint(
            "scheduled_date",
            "scheduled_time",
            name="uq_interview_booking_slot",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(index=True)
    name: Mapped[str] = mapped_column(String(150))
    email: Mapped[str] = mapped_column(String(320), index=True)
    scheduled_date: Mapped[date] = mapped_column(Date)
    scheduled_time: Mapped[time] = mapped_column(Time)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
