"""Environment-based application settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables and an optional .env file."""

    app_name: str = "RAG Backend"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, validation_alias="APP_DEBUG")
    api_v1_prefix: str = "/api/v1"
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@127.0.0.1:5433/rag_backend"
    )
    max_upload_size_mb: int = 10
    recursive_chunk_size: int = 1500
    recursive_chunk_overlap: int = 200
    semantic_breakpoint_percentile: float = 80.0
    semantic_min_chunk_size: int = 400
    semantic_max_chunk_size: int = 1500
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "embeddinggemma"
    embedding_batch_size: int = 32
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "document_chunks"
    chat_model: str = "gemma3:1b"
    chat_temperature: float = 0.1
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.2
    redis_url: str = "redis://127.0.0.1:6379/0"
    chat_memory_ttl_seconds: int = 86400
    chat_memory_max_messages: int = 12

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Build settings once and reuse them throughout the process."""
    return Settings()


settings = get_settings()
