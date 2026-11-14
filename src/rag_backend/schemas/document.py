"""Document API schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from rag_backend.models.document import ChunkingStrategy, DocumentStatus


class DocumentResponse(BaseModel):
    """Public representation of an ingested document."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    content_type: str
    file_size: int
    chunking_strategy: ChunkingStrategy
    status: DocumentStatus
    error_message: str | None
    created_at: datetime
    updated_at: datetime
