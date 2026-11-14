"""Unit tests for custom chat orchestration."""

import json
import uuid
from datetime import date, datetime, time, timezone
from types import SimpleNamespace

import pytest

from rag_backend.integrations.llm.base import ChatMessage
from rag_backend.integrations.vector_store.base import VectorSearchResult
from rag_backend.services.chat import ChatService


class FakeEmbeddingProvider:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def close(self) -> None: ...


class FakeVectorStore:
    async def search(self, query_vector, *, limit, score_threshold=None):
        return [
            VectorSearchResult(
                chunk_id="chunk-1",
                document_id="document-1",
                filename="guide.txt",
                chunk_index=0,
                page_number=1,
                text="FastAPI is a Python API framework.",
                score=0.9,
            )
        ]

    async def upsert_document(self, document, chunks, embeddings): ...
    async def close(self) -> None: ...


class FakeChatModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def generate(self, messages, *, response_schema=None) -> str:
        return self.responses.pop(0)

    async def close(self) -> None: ...


class FakeMemory:
    def __init__(self) -> None:
        self.messages: dict[uuid.UUID, list[ChatMessage]] = {}

    async def get(self, conversation_id: uuid.UUID) -> list[ChatMessage]:
        return self.messages.get(conversation_id, [])

    async def append(self, conversation_id, messages) -> None:
        self.messages.setdefault(conversation_id, []).extend(messages)

    async def close(self) -> None: ...


class FakeBookingRepository:
    async def create(self, **values):
        return SimpleNamespace(
            id=uuid.uuid4(),
            created_at=datetime.now(timezone.utc),
            **values,
        )


def build_service(model: FakeChatModel, memory: FakeMemory) -> ChatService:
    return ChatService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        chat_model=model,
        memory=memory,
        booking_repository=FakeBookingRepository(),
        retrieval_top_k=5,
        retrieval_score_threshold=0.2,
    )


@pytest.mark.asyncio
async def test_answers_with_retrieved_sources_and_remembers_turn() -> None:
    memory = FakeMemory()
    model = FakeChatModel(
        [
            json.dumps({"booking_intent": False}),
            "FastAPI is a Python API framework [Source 1].",
        ]
    )

    result = await build_service(model, memory).respond(
        message="What is FastAPI?",
        conversation_id=None,
    )

    assert result.action == "answer"
    assert result.sources[0].filename == "guide.txt"
    assert len(memory.messages[result.conversation_id]) == 2


@pytest.mark.asyncio
async def test_requests_missing_booking_details() -> None:
    memory = FakeMemory()
    model = FakeChatModel(
        [json.dumps({"booking_intent": True, "name": "Sandip"})]
    )

    result = await build_service(model, memory).respond(
        message="Book an interview for me",
        conversation_id=None,
    )

    assert result.action == "booking_details_required"
    assert "email" in result.answer
    assert "scheduled_date" in result.answer


@pytest.mark.asyncio
async def test_persists_complete_booking() -> None:
    memory = FakeMemory()
    model = FakeChatModel(
        [
            json.dumps(
                {
                    "booking_intent": True,
                    "name": "Sandip Kumar",
                    "email": "sandip@example.com",
                    "date": "2099-01-15",
                    "time": "10:30:00",
                }
            )
        ]
    )

    result = await build_service(model, memory).respond(
        message="Book my interview",
        conversation_id=None,
    )

    assert result.action == "interview_booked"
    assert result.booking.scheduled_date == date(2099, 1, 15)
    assert result.booking.scheduled_time == time(10, 30)
