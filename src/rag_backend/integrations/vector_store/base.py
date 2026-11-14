"""Vector-store contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_backend.models.document import Document
from rag_backend.models.document_chunk import DocumentChunk


class VectorStoreError(RuntimeError):
    """Raised when vectors cannot be persisted or queried."""


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """One document chunk returned by vector similarity search."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    text: str
    score: float


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

    async def search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Return chunks nearest to a query embedding."""
        ...

    async def close(self) -> None:
        """Release vector-store resources."""
        ...
