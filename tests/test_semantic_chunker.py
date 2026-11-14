"""Focused tests for semantic topic-boundary chunking."""

import pytest

from rag_backend.chunkers.semantic import SemanticChunker
from rag_backend.extractors.base import ExtractedDocument, ExtractedPage


class TopicEmbeddingProvider:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "Cat" in text or "pet" in text else [0.0, 1.0]
            for text in texts
        ]

    async def close(self) -> None: ...


@pytest.mark.asyncio
async def test_splits_at_a_semantic_topic_change() -> None:
    chunker = SemanticChunker(
        embedding_provider=TopicEmbeddingProvider(),
        breakpoint_percentile=80,
        min_chunk_size=10,
        max_chunk_size=200,
    )
    document = ExtractedDocument(
        pages=(
            ExtractedPage(
                "Cats are pets. A pet likes warmth. SQL stores records. "
                "Databases support queries.",
                page_number=3,
            ),
        )
    )

    chunks = await chunker.split(document)

    assert len(chunks) == 2
    assert chunks[0].page_number == 3
    assert "Cats" in chunks[0].text
    assert "SQL" in chunks[1].text
