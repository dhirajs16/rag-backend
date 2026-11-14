"""Conversational RAG and interview-booking endpoint."""

from typing import Literal, cast

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.config import settings
from rag_backend.db.session import get_db_session
from rag_backend.integrations.embeddings import OllamaEmbeddingProvider
from rag_backend.integrations.embeddings.base import EmbeddingError
from rag_backend.integrations.llm import LLMError, OllamaChatModel
from rag_backend.integrations.memory import MemoryError, RedisConversationMemory
from rag_backend.integrations.vector_store import QdrantVectorStore
from rag_backend.integrations.vector_store.base import VectorStoreError
from rag_backend.repositories.interview_booking import (
    BookingConflictError,
    InterviewBookingRepository,
)
from rag_backend.schemas.chat import (
    BookingResponse,
    ChatRequest,
    ChatResponse,
    ChatSourceResponse,
)
from rag_backend.services.chat import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ChatResponse:
    """Answer from documents or collect and persist an interview booking."""
    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    chat_model = OllamaChatModel(
        base_url=settings.ollama_url,
        model=settings.chat_model,
        temperature=settings.chat_temperature,
    )
    memory = RedisConversationMemory(
        url=settings.redis_url,
        ttl_seconds=settings.chat_memory_ttl_seconds,
        max_messages=settings.chat_memory_max_messages,
    )
    service = ChatService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_model=chat_model,
        memory=memory,
        booking_repository=InterviewBookingRepository(session),
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_score_threshold=settings.retrieval_score_threshold,
    )

    try:
        result = await service.respond(
            message=request.message,
            conversation_id=request.conversation_id,
        )
    except BookingConflictError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
    except (EmbeddingError, VectorStoreError, LLMError, MemoryError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    finally:
        await embedding_provider.close()
        await vector_store.close()
        await chat_model.close()
        await memory.close()

    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        action=cast(
            Literal["answer", "booking_details_required", "interview_booked"],
            result.action,
        ),
        sources=[
            ChatSourceResponse(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                filename=source.filename,
                page_number=source.page_number,
                score=source.score,
            )
            for source in result.sources
        ],
        booking=(
            BookingResponse.model_validate(result.booking, from_attributes=True)
            if result.booking is not None
            else None
        ),
    )
