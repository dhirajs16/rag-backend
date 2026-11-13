"""FastAPI application factory and entry point."""

from fastapi import FastAPI

from rag_backend.api.router import api_router
from rag_backend.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_app()

