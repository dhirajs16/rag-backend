"""Document-ingestion endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.config import settings
from rag_backend.db.session import get_db_session
from rag_backend.integrations.embeddings import OllamaEmbeddingProvider
from rag_backend.integrations.vector_store import QdrantVectorStore
from rag_backend.models.document import ChunkingStrategy
from rag_backend.repositories.document import DocumentRepository
from rag_backend.schemas.document import DocumentResponse
from rag_backend.services.ingestion import (
    ChunkingUnavailableError,
    DocumentTooLargeError,
    DuplicateDocumentError,
    IngestionProcessingError,
    IngestionService,
    InvalidDocumentError,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(description="A PDF or plain-text document")],
    chunking_strategy: Annotated[ChunkingStrategy, Form()],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    """Validate a document, extract its text, and persist its metadata."""
    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    service = IngestionService(
        DocumentRepository(session),
        embedding_provider,
        vector_store,
    )

    try:
        document = await service.ingest(file, chunking_strategy)
    except InvalidDocumentError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
    except DocumentTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        )
    except ChunkingUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(error),
        )
    except DuplicateDocumentError as error:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
    except IngestionProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        )
    finally:
        await embedding_provider.close()
        await vector_store.close()

    return DocumentResponse.model_validate(document)
