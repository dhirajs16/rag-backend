"""Database operations for documents."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.chunkers.base import TextChunk
from rag_backend.models.document import ChunkingStrategy, Document
from rag_backend.models.document_chunk import DocumentChunk


class DocumentRepository:
    """Persist and query document metadata."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_checksum(self, checksum: str) -> Document | None:
        """Return a document with the supplied checksum, when one exists."""
        result = await self._session.execute(
            select(Document).where(Document.checksum == checksum)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        original_filename: str,
        content_type: str,
        file_size: int,
        checksum: str,
        chunking_strategy: ChunkingStrategy,
        chunks: Sequence[TextChunk],
    ) -> Document:
        """Create a document and all of its chunks in one transaction."""
        document = Document(
            original_filename=original_filename,
            content_type=content_type,
            file_size=file_size,
            checksum=checksum,
            chunking_strategy=chunking_strategy,
            chunks=[
                DocumentChunk(
                    chunk_index=chunk.index,
                    text=chunk.text,
                    character_count=len(chunk.text),
                    page_number=chunk.page_number,
                )
                for chunk in chunks
            ],
        )
        self._session.add(document)
        await self._session.commit()
        await self._session.refresh(document)
        return document
