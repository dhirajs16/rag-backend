"""Document retrieval endpoints."""

from fastapi import APIRouter, HTTPException, status

from rag_backend.core.config import settings
from rag_backend.integrations.embeddings import OllamaEmbeddingProvider
from rag_backend.integrations.embeddings.base import EmbeddingError
from rag_backend.integrations.vector_store import QdrantVectorStore
from rag_backend.integrations.vector_store.base import VectorStoreError
from rag_backend.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkResponse,
)
from rag_backend.services.retrieval import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post("/search", response_model=RetrievalResponse)
async def search_documents(request: RetrievalRequest) -> RetrievalResponse:
    """Retrieve document chunks that are semantically related to a query."""
    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    service = RetrievalService(embedding_provider, vector_store)

    try:
        matches = await service.search(
            request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except (EmbeddingError, VectorStoreError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    finally:
        await embedding_provider.close()
        await vector_store.close()

    return RetrievalResponse(
        query=request.query,
        matches=[
            RetrievedChunkResponse.model_validate(match, from_attributes=True)
            for match in matches
        ],
    )
