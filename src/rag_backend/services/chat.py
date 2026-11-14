"""Custom conversational RAG and interview-booking orchestration."""

import json
import re
import uuid
from datetime import datetime

from pydantic import ValidationError

from rag_backend.integrations.embeddings.base import EmbeddingProvider
from rag_backend.integrations.llm.base import ChatMessage, ChatModel, LLMError
from rag_backend.integrations.memory.base import ConversationMemory
from rag_backend.integrations.vector_store.base import VectorSearchResult, VectorStore
from rag_backend.models.interview_booking import InterviewBooking
from rag_backend.repositories.interview_booking import InterviewBookingRepository
from rag_backend.schemas.chat import BookingIntentExtraction, ValidatedBookingDetails


class ChatResult:
    """Internal result returned by the chat orchestration service."""

    def __init__(
        self,
        *,
        conversation_id: uuid.UUID,
        answer: str,
        action: str,
        sources: list[VectorSearchResult] | None = None,
        booking: InterviewBooking | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.answer = answer
        self.action = action
        self.sources = sources or []
        self.booking = booking


class ChatService:
    """Run retrieval, prompt generation, memory, and booking workflows."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chat_model: ChatModel,
        memory: ConversationMemory,
        booking_repository: InterviewBookingRepository,
        retrieval_top_k: int,
        retrieval_score_threshold: float,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chat_model = chat_model
        self._memory = memory
        self._booking_repository = booking_repository
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_score_threshold = retrieval_score_threshold

    async def respond(
        self,
        *,
        message: str,
        conversation_id: uuid.UUID | None,
    ) -> ChatResult:
        conversation_id = conversation_id or uuid.uuid4()
        history = await self._memory.get(conversation_id)
        booking_intent = await self._extract_booking_intent(history, message)
        booking_intent = self._supplement_booking_extraction(
            booking_intent,
            history,
            message,
        )

        if booking_intent.booking_intent:
            return await self._handle_booking(
                conversation_id,
                history,
                message,
                booking_intent,
            )

        query = self._build_retrieval_query(history, message)
        query_vector = (await self._embedding_provider.embed_texts([query]))[0]
        sources = await self._vector_store.search(
            query_vector,
            limit=self._retrieval_top_k,
            score_threshold=self._retrieval_score_threshold,
        )
        answer = await self._generate_grounded_answer(history, message, sources)
        await self._remember(conversation_id, message, answer)
        return ChatResult(
            conversation_id=conversation_id,
            answer=answer,
            action="answer",
            sources=sources,
        )

    async def _extract_booking_intent(
        self,
        history: list[ChatMessage],
        message: str,
    ) -> BookingIntentExtraction:
        transcript = self._format_history(history)
        prompt = (
            "Determine whether the user is trying to book an interview. Extract "
            "name, email, date, and time from the entire conversation. Return null "
            "for anything not provided. Resolve relative dates using today, "
            f"{datetime.now().date().isoformat()}. Use YYYY-MM-DD for date and "
            "HH:MM:SS for 24-hour time.\n\n"
            f"Conversation history:\n{transcript or '(none)'}\n"
            f"Latest user message:\n{message}"
        )
        raw_response = await self._chat_model.generate(
            [ChatMessage(role="user", content=prompt)],
            response_schema=BookingIntentExtraction.model_json_schema(),
        )
        try:
            return BookingIntentExtraction.model_validate_json(raw_response)
        except ValidationError as error:
            raise LLMError("The model returned invalid booking-intent data.") from error

    async def _handle_booking(
        self,
        conversation_id: uuid.UUID,
        history: list[ChatMessage],
        message: str,
        extraction: BookingIntentExtraction,
    ) -> ChatResult:
        raw_details = {
            "name": extraction.name,
            "email": extraction.email,
            "scheduled_date": extraction.date,
            "scheduled_time": extraction.time,
        }
        missing = [name for name, value in raw_details.items() if not value]
        try:
            details = ValidatedBookingDetails.model_validate(raw_details)
        except ValidationError as error:
            invalid_fields = [str(item["loc"][0]) for item in error.errors()]
            missing.extend(invalid_fields)
            details = None

        if details is not None:
            scheduled_at = datetime.combine(
                details.scheduled_date,
                details.scheduled_time,
            )
            if scheduled_at <= datetime.now():
                missing.append("future date and time")
                details = None

        if details is None:
            requested = ", ".join(dict.fromkeys(missing))
            answer = f"Please provide valid interview details for: {requested}."
            await self._remember(conversation_id, message, answer)
            return ChatResult(
                conversation_id=conversation_id,
                answer=answer,
                action="booking_details_required",
            )

        booking = await self._booking_repository.create(
            conversation_id=conversation_id,
            name=details.name,
            email=str(details.email),
            scheduled_date=details.scheduled_date,
            scheduled_time=details.scheduled_time,
        )
        answer = (
            f"Interview booked for {booking.name} on "
            f"{booking.scheduled_date.isoformat()} at "
            f"{booking.scheduled_time.strftime('%H:%M')}."
        )
        await self._remember(conversation_id, message, answer)
        return ChatResult(
            conversation_id=conversation_id,
            answer=answer,
            action="interview_booked",
            booking=booking,
        )

    @staticmethod
    def _supplement_booking_extraction(
        extraction: BookingIntentExtraction,
        history: list[ChatMessage],
        message: str,
    ) -> BookingIntentExtraction:
        """Recover explicit fields a small model occasionally overlooks."""
        text = "\n".join([*(item.content for item in history), message])
        email_match = re.search(
            r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
            text,
            flags=re.IGNORECASE,
        )
        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        time_match = re.search(
            r"\b(0?[1-9]|1[0-2]):([0-5]\d)(?:\s*([AP]M))\b",
            text,
            flags=re.IGNORECASE,
        )
        name_match = re.search(
            r"(?:book(?:\s+an)?\s+interview\s+for|name(?:\s+is)?)[ :]"
            r"*([A-Za-z][A-Za-z .'-]{1,149}?)(?=,|\s+email\b|$)",
            text,
            flags=re.IGNORECASE,
        )

        extracted_time: str | None = None
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            meridiem = time_match.group(3).upper()
            if meridiem == "PM" and hour != 12:
                hour += 12
            elif meridiem == "AM" and hour == 12:
                hour = 0
            extracted_time = f"{hour:02d}:{minute:02d}:00"

        booking_words = "book" in text.lower() and "interview" in text.lower()
        return extraction.model_copy(
            update={
                "booking_intent": extraction.booking_intent or booking_words,
                "name": extraction.name or (name_match.group(1).strip() if name_match else None),
                "email": extraction.email or (email_match.group(0) if email_match else None),
                "date": extraction.date or (date_match.group(0) if date_match else None),
                "time": extraction.time or extracted_time,
            }
        )

    async def _generate_grounded_answer(
        self,
        history: list[ChatMessage],
        message: str,
        sources: list[VectorSearchResult],
    ) -> str:
        context = "\n\n".join(
            f"[Source {index}: {source.filename}, page "
            f"{source.page_number or 'unknown'}]\n{source.text}"
            for index, source in enumerate(sources, start=1)
        )
        system_message = ChatMessage(
            role="system",
            content=(
                "You are a document question-answering assistant. Answer only from "
                "the supplied document context. Use conversation history to resolve "
                "follow-up references. If the context is insufficient, say that the "
                "uploaded documents do not contain enough information. Cite supporting "
                "passages as [Source N]. Never invent facts or booking confirmations."
            ),
        )
        user_message = ChatMessage(
            role="user",
            content=(
                f"Document context:\n{context or '(no relevant context found)'}\n\n"
                f"Question:\n{message}"
            ),
        )
        return await self._chat_model.generate(
            [system_message, *history, user_message]
        )

    @staticmethod
    def _build_retrieval_query(history: list[ChatMessage], message: str) -> str:
        recent_user_messages = [
            item.content for item in history if item.role == "user"
        ][-2:]
        return "\n".join([*recent_user_messages, message])

    @staticmethod
    def _format_history(history: list[ChatMessage]) -> str:
        return "\n".join(f"{item.role}: {item.content}" for item in history)

    async def _remember(
        self,
        conversation_id: uuid.UUID,
        user_message: str,
        assistant_message: str,
    ) -> None:
        await self._memory.append(
            conversation_id,
            [
                ChatMessage(role="user", content=user_message),
                ChatMessage(role="assistant", content=assistant_message),
            ],
        )
