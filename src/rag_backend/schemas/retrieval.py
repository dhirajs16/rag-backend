"""Retrieval API request and response schemas."""

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """A natural-language query and its retrieval controls."""

    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float | None = Field(default=0.3, ge=-1.0, le=1.0)


class RetrievedChunkResponse(BaseModel):
    """A relevant source chunk returned from the vector store."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    text: str
    score: float


class RetrievalResponse(BaseModel):
    """Ranked source chunks for one query."""

    query: str
    matches: list[RetrievedChunkResponse]
