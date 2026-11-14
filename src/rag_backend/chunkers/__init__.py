"""Document chunker selection."""

from rag_backend.chunkers.base import (
    DocumentChunker,
    TextChunk,
)
from rag_backend.chunkers.recursive import RecursiveCharacterChunker
from rag_backend.chunkers.semantic import SemanticChunker
from rag_backend.integrations.embeddings.base import EmbeddingProvider
from rag_backend.models.document import ChunkingStrategy


def get_chunker(
    strategy: ChunkingStrategy,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_provider: EmbeddingProvider,
    semantic_breakpoint_percentile: float,
    semantic_min_chunk_size: int,
    semantic_max_chunk_size: int,
) -> DocumentChunker:
    """Build the chunker configured for the requested strategy."""
    if strategy is ChunkingStrategy.RECURSIVE:
        return RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return SemanticChunker(
        embedding_provider=embedding_provider,
        breakpoint_percentile=semantic_breakpoint_percentile,
        min_chunk_size=semantic_min_chunk_size,
        max_chunk_size=semantic_max_chunk_size,
    )


__all__ = ["TextChunk", "get_chunker"]
