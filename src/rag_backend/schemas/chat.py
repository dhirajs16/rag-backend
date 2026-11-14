"""Conversational RAG API schemas."""

import uuid
from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class ChatRequest(BaseModel):
    """One user turn, optionally continuing an existing conversation."""

    message: str = Field(min_length=2, max_length=4000)
    conversation_id: uuid.UUID | None = None


class ChatSourceResponse(BaseModel):
    """Document evidence used to ground an answer."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None
    score: float


class BookingResponse(BaseModel):
    """A successfully persisted interview booking."""

    id: uuid.UUID
    name: str
    email: EmailStr
    scheduled_date: date
    scheduled_time: time
    created_at: datetime


class ChatResponse(BaseModel):
    """A grounded answer or an interview-booking result."""

    conversation_id: uuid.UUID
    answer: str
    action: Literal["answer", "booking_details_required", "interview_booked"]
    sources: list[ChatSourceResponse] = Field(default_factory=list)
    booking: BookingResponse | None = None


class BookingIntentExtraction(BaseModel):
    """Loose LLM output; definitive validation happens separately."""

    booking_intent: bool
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class ValidatedBookingDetails(BaseModel):
    """Strict values required before a booking reaches PostgreSQL."""

    name: str = Field(min_length=2, max_length=150)
    email: EmailStr
    scheduled_date: date
    scheduled_time: time
