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
