"""Interview booking database operations."""

import uuid
from datetime import date, time

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.models.interview_booking import InterviewBooking


class BookingConflictError(RuntimeError):
    """Raised when an interview slot has already been reserved."""


class InterviewBookingRepository:
    """Persist confirmed interview bookings."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        conversation_id: uuid.UUID,
        name: str,
        email: str,
        scheduled_date: date,
        scheduled_time: time,
    ) -> InterviewBooking:
        booking = InterviewBooking(
            conversation_id=conversation_id,
            name=name,
            email=email,
            scheduled_date=scheduled_date,
            scheduled_time=scheduled_time,
        )
        self._session.add(booking)
        try:
            await self._session.commit()
        except IntegrityError as error:
            await self._session.rollback()
            raise BookingConflictError(
                "That interview date and time has already been booked."
            ) from error
        await self._session.refresh(booking)
        return booking
