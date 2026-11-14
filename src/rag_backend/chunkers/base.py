"""Shared types and contracts for text chunking."""

from dataclasses import dataclass
from typing import Protocol

from rag_backend.extractors.base import ExtractedDocument


class ChunkingError(ValueError):
    """Raised when a document cannot be split with a requested strategy."""


class ChunkingStrategyUnavailableError(ChunkingError):
    """Raised when a known chunking strategy is not implemented yet."""


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One retrieval unit produced from an extracted document."""

    index: int
    text: str
    page_number: int | None


class DocumentChunker(Protocol):
    """Contract implemented by document chunking strategies."""

    async def split(self, document: ExtractedDocument) -> list[TextChunk]:
        """Split an extracted document into ordered retrieval units."""
        ...
