"""Document-ingestion orchestration."""

import hashlib
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError

from rag_backend.core.config import settings
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


class IngestionService:
    """Validate uploads and persist their metadata."""

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def ingest(
        self,
        file: UploadFile,
        chunking_strategy: ChunkingStrategy,
    ) -> Document:
        """Validate an upload, calculate its checksum, and persist metadata."""
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

        try:
            return await self._repository.create(
                original_filename=filename,
                content_type=content_type,
                file_size=len(content),
                checksum=checksum,
                chunking_strategy=chunking_strategy,
            )
        except IntegrityError as error:
            raise DuplicateDocumentError(
                "This document has already been uploaded."
            ) from error
