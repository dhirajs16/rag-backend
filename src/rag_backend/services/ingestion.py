"""Document-ingestion orchestration."""

import asyncio
import hashlib
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError

from rag_backend.chunkers import get_chunker
from rag_backend.chunkers.base import ChunkingStrategyUnavailableError
from rag_backend.core.config import settings
from rag_backend.extractors import get_extractor
from rag_backend.extractors.base import ExtractionError
from rag_backend.integrations.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_backend.integrations.vector_store.base import VectorStore, VectorStoreError
from rag_backend.models.document import ChunkingStrategy, Document
from rag_backend.repositories.document import DocumentRepository

ALLOWED_CONTENT_TYPES: dict[str, set[str]] = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
}


class IngestionError(Exception):
    """Base error for an expected ingestion failure."""


class InvalidDocumentError(IngestionError):
    """Raised when the uploaded document is empty or unsupported."""


class DocumentTooLargeError(IngestionError):
    """Raised when the upload exceeds the configured size limit."""


class DuplicateDocumentError(IngestionError):
    """Raised when the same file content was previously uploaded."""


class ChunkingUnavailableError(IngestionError):
    """Raised when the requested chunking strategy is not ready for use."""


class IngestionProcessingError(IngestionError):
    """Raised when embeddings or vector persistence fail."""


class IngestionService:
    """Validate uploads, extract text, and persist their metadata."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def ingest(
        self,
        file: UploadFile,
        chunking_strategy: ChunkingStrategy,
    ) -> Document:
        """Validate an upload, extract its text, and persist metadata."""
        filename = Path(file.filename or "").name
        if not filename:
            raise InvalidDocumentError("A filename is required.")

        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_CONTENT_TYPES:
            raise InvalidDocumentError("Only .pdf and .txt files are supported.")

        content_type = file.content_type or "application/octet-stream"
        if content_type not in ALLOWED_CONTENT_TYPES[extension]:
            raise InvalidDocumentError(
                f"Content type '{content_type}' does not match a {extension} file."
            )

        maximum_bytes = settings.max_upload_size_mb * 1024 * 1024
        content = await file.read(maximum_bytes + 1)
        await file.close()

        if not content:
            raise InvalidDocumentError("The uploaded file is empty.")
        if len(content) > maximum_bytes:
            raise DocumentTooLargeError(
                f"The file exceeds the {settings.max_upload_size_mb} MB limit."
            )

        checksum = hashlib.sha256(content).hexdigest()
        if await self._repository.get_by_checksum(checksum) is not None:
            raise DuplicateDocumentError("This document has already been uploaded.")

        extractor = get_extractor(extension)
        try:
            extracted_document = await asyncio.to_thread(extractor.extract, content)
        except ExtractionError as error:
            raise InvalidDocumentError(str(error)) from error

        try:
            chunker = get_chunker(
                chunking_strategy,
                chunk_size=settings.recursive_chunk_size,
                chunk_overlap=settings.recursive_chunk_overlap,
            )
            chunks = await asyncio.to_thread(chunker.split, extracted_document)
        except ChunkingStrategyUnavailableError as error:
            raise ChunkingUnavailableError(str(error)) from error

        try:
            document, stored_chunks = await self._repository.create(
                original_filename=filename,
                content_type=content_type,
                file_size=len(content),
                checksum=checksum,
                chunking_strategy=chunking_strategy,
                chunks=chunks,
            )
        except IntegrityError as error:
            raise DuplicateDocumentError(
                "This document has already been uploaded."
            ) from error

        try:
            embeddings = await self._embedding_provider.embed_texts(
                [chunk.text for chunk in stored_chunks]
            )
            await self._vector_store.upsert_document(
                document,
                stored_chunks,
                embeddings,
            )
            return await self._repository.mark_completed(document)
        except (EmbeddingError, VectorStoreError) as error:
            await self._repository.mark_failed(document, str(error))
            raise IngestionProcessingError(str(error)) from error
