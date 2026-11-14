"""Document chunker selection."""

from rag_backend.chunkers.base import (
    ChunkingStrategyUnavailableError,
    DocumentChunker,
    TextChunk,
)
from rag_backend.chunkers.recursive import RecursiveCharacterChunker
from rag_backend.models.document import ChunkingStrategy


def get_chunker(
    strategy: ChunkingStrategy,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> DocumentChunker:
    """Build the chunker configured for the requested strategy."""
    if strategy is ChunkingStrategy.RECURSIVE:
        return RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise ChunkingStrategyUnavailableError(
        "Semantic chunking will be enabled after the embedding provider is added."
    )


__all__ = ["TextChunk", "get_chunker"]

