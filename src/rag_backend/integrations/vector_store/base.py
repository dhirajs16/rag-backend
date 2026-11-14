"""Vector-store contracts."""

from collections.abc import Sequence
from typing import Protocol

from rag_backend.models.document import Document
from rag_backend.models.document_chunk import DocumentChunk


class VectorStoreError(RuntimeError):
    """Raised when vectors cannot be persisted or queried."""


class VectorStore(Protocol):
    """Contract for persistent vector storage."""

    async def upsert_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store all embedded chunks belonging to one document."""
        ...

    async def close(self) -> None:
        """Release vector-store resources."""
        ...

